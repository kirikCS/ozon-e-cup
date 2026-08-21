"""Ствол Toto 2.0 с нашей генеративной головой — спасает ли предобучение.

Zero-shot провалился структурно: P(Y>0) = 0.9967 против эмпирических 0.5830, Toto практически
никогда не предсказывает ноль. Обучен на телеметрии наблюдаемости, где нулей не бывает, и
никакая копула этого не чинит — она перераспределяет массу между днями, но не создаёт нулей там,
где их нет в маргиналах.

Остаётся вопрос, отделима ли беда от ствола. Квантильная голова обучена на чужом домене; сам
трансформер мог выучить полезное представление формы ряда. Проверяется прямо: снимаем скрытое
состояние хуком, приделываем НАШУ генеративную голову (30 × хазард, μ, log σ) и учим её на наших
якорях, оставив ствол замороженным.

Рука frozen отвечает на вопрос «полезно ли представление». Если она проиграет нашей свёрточной
модели заметно, то предобучение на этом домене бесполезно, и дообучение целиком (много дороже)
ставить незачем.

Контроль обязателен: та же голова на СЛУЧАЙНОМ проекции входа вместо ствола Toto. Без него
нельзя отличить «ствол полезен» от «голова сама по себе неплоха».
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from toto2 import Toto2Model

DEV = "cuda"
CTX = 320
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS_EMB = int(os.environ.get("BS_EMB", 128))
EPOCHS, BS, LR = 8, 512, 2e-3
NSAMP, GH, SIGMA = 96, 16, 0.8
MODEL = os.environ.get("TOTO_MODEL", "Datadog/Toto-2.0-22m")
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)

def win(users, end, chans):
    lo = end - CTX + 1
    if lo >= 0:
        return np.stack([M[users, lo:end + 1] for M in chans], 1).astype(np.float32)
    got = np.stack([M[users, 0:end + 1] for M in chans], 1).astype(np.float32)
    return np.concatenate([np.zeros((len(users), len(chans), -lo), np.float32), got], 2)

print(f"загружаю {MODEL} ...", flush=True)
toto = Toto2Model.from_pretrained(MODEL).to(DEV).eval()
for p in toto.parameters():
    p.requires_grad_(False)
print(f"  {sum(p.numel() for p in toto.parameters())/1e6:.1f}М параметров, ствол заморожен",
      flush=True)

CAP = {}
def hook(_m, _i, o):
    CAP["h"] = o[0] if isinstance(o, (tuple, list)) else o
toto.transformer.register_forward_hook(hook)

CH = [G, V, C]
@torch.no_grad()
def embed(anchor):
    outs = []
    for i in range(0, NU, BS_EMB):
        u = np.arange(i, min(i + BS_EMB, NU))
        x = torch.from_numpy(win(u, anchor, CH)).to(DEV)
        msk = torch.ones_like(x, dtype=torch.bool)
        sid = torch.zeros(len(u), len(CH), dtype=torch.long, device=DEV)
        toto(target=x, target_mask=msk, cpm_mask=torch.ones_like(msk), series_ids=sid)
        h = CAP["h"]
        if i == 0:
            print(f"    скрытое состояние: {tuple(h.shape)}", flush=True)
        # (batch, variate, patch, dim). Разворачивать целиком нельзя: 3*10*512 = 15360 признаков
        # на клиента, то есть 15 ГБ на якорь при 36 ГБ памяти — первая версия так и умерла от
        # OOM-killer молча. Сворачиваем патчи в среднее и последний, берём целевую переменную.
        if h.dim() == 4:
            h = torch.cat([h[:, 0].mean(1), h[:, 0, -1]], -1)
        else:
            h = h.flatten(1)
        outs.append(h.float().cpu().numpy())
    return np.concatenate(outs, 0)

print("снимаю представления ...", flush=True)
t0 = time.time()
EMB = {}
for a in SEL_TR + [SEL_EV]:
    EMB[a] = embed(a)
    print(f"  якорь {a}: {EMB[a].shape}  ({time.time()-t0:.0f}s)", flush=True)
D = EMB[SEL_EV].shape[1]
mu_ = np.mean([EMB[a].mean(0) for a in SEL_TR], 0)
sd_ = np.maximum(np.mean([EMB[a].std(0) for a in SEL_TR], 0), 1e-3)
for a in EMB:
    EMB[a] = np.clip((EMB[a] - mu_) / sd_, -8, 8).astype(np.float32)

class Head(nn.Module):
    def __init__(self, din, d=192):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(din, 512), nn.SiLU(), nn.Linear(512, d), nn.SiLU())
        self.pos = nn.Parameter(torch.randn(HOR, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(d + 16, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, e):
        h = self.body(e)
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, HOR, -1),
                       self.pos.unsqueeze(0).expand(B, HOR, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); lyd = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyd - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(tag, feats, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Head(feats[SEL_EV].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64: continue
                e = torch.from_numpy(feats[a][idx]).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(e), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    m.eval()
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 1024):
            idx = np.arange(i, min(i + 1024, NU))
            hz, mu, ls = m(torch.from_numpy(feats[SEL_EV][idx]).to(DEV))
            hz = hz * scale; sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NSAMP, len(idx), HOR, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NSAMP, len(idx), HOR, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out

y = target(SEL_EV); ly = np.log1p(y)
ref = OUT / "h8_gauss.npy"
if ref.exists():
    r = np.load(ref); r = r - r.mean() + ly.mean()
    print(f"\nнаша свёрточная дневная модель: {rmsle(y, np.expm1(np.clip(r,0,13))):.5f}", flush=True)

print(f"\n{'рука':>14} {'признаков':>10} {'RMSLE':>9} {'сек':>7}", flush=True)
RES = {}
t0 = time.time()
v = run("toto_frozen", EMB)
RES["toto_frozen"] = v
sh = v - v.mean() + ly.mean()
print(f"{'toto_frozen':>14} {D:10d} {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
      f"{time.time()-t0:7.0f}", flush=True)
np.save(OUT / "k3_toto_frozen.npy", v)

# КОНТРОЛЬ: та же голова на случайной проекции сырого окна той же размерности
rs = np.random.default_rng(0)
W = rs.standard_normal((len(CH) * CTX, min(D, 1024))).astype(np.float32) / np.sqrt(len(CH) * CTX)
RND = {}
for a in SEL_TR + [SEL_EV]:
    X = win(np.arange(NU), a, CH).reshape(NU, -1)
    X = np.log1p(np.maximum(X, 0))
    RND[a] = (X @ W).astype(np.float32)
mu2 = np.mean([RND[a].mean(0) for a in SEL_TR], 0)
sd2 = np.maximum(np.mean([RND[a].std(0) for a in SEL_TR], 0), 1e-3)
for a in RND:
    RND[a] = np.clip((RND[a] - mu2) / sd2, -8, 8).astype(np.float32)
t0 = time.time()
v2 = run("random_proj", RND)
sh2 = v2 - v2.mean() + ly.mean()
print(f"{'random_proj':>14} {RND[SEL_EV].shape[1]:10d} "
      f"{rmsle(y, np.expm1(np.clip(sh2,0,13))):9.5f} {time.time()-t0:7.0f}", flush=True)
print(f"\ncorr(toto_frozen, random_proj) = {np.corrcoef(v, v2)[0,1]:.5f}")
if ref.exists():
    print(f"corr(toto_frozen, наша дневная) = {np.corrcoef(v, np.load(ref))[0,1]:.5f}")
print("\nчитать так: если toto_frozen не бьёт random_proj — предобученное представление")
print("не несёт ничего сверх произвольной линейной сводки того же окна")
print("\ndone")
