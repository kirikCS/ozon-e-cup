"""PatchTST-FM: новая ось предобучения и 99-квантильная голова с атомом в нуле.

Два вопроса, и они разные, поэтому дизайн 2x2.

ОСЬ ПРЕДОБУЧЕНИЯ. Все стволы, что мы зондировали (Toto, Chronos, наша диффузия), обучены
прогнозом. PatchTST-FM обучен масочной реконструкцией: прогноз при выводе оформлен как
восстановление замаскированного будущего при незамаскированном прошлом. Это другая ось, а не
четвёртая копия опыта. И лежит она на 258 млн — в полосе, где наша кривая масштаба развернулась
(Chronos 119,5М проиграл контролю, Toto 313М обошёл его на 0,010).

99-КВАНТИЛЬНАЯ ГОЛОВА. Каждый провал квантильных моделей у нас читался как «атома нет»: при
P(Y>0)=0,583 Toto давал 0,997, Chronos 0,944, диффузия ровно 1,000. Но 99 квантилей атом
представить МОГУТ — нижние ~41 схлопываются в ноль. Голова учится на месячную сумму НАПРЯМУЮ,
поэтому джессеновский разрыв (измеренный 1,0906) не чинится квадратурой, а не возникает вовсе.
Считаем квантили L=log1p(Y): тогда E[L] = средняя по квантилям, что и есть точный оптимум RMSLE
`\\hat y = expm1(E[log1p y|x])`. Монотонность берём тем же приёмом, что у самой PatchTST-FM:
база плюс кумулятивные softplus-приращения.

Контроль обязателен и той же формы (у Toto подтвердил сигнал, у Chronos снял его): случайная
линейная проекция сырого окна той же размерности через ту же голову. Обе головы гоняются на обоих
стволах — иначе «ствол помог» и «голова хороша» неразличимы.

Отвод: хук на вход out_layer даёт (B, 28, 1024) до квантильной проекции. Подготовку входа
(паддинг, маски, инстанс-норму) отдаём библиотеке — соглашение сложное, угадывать нечего.
Урок k3->k4: патчи СОХРАНЯЮТСЯ, жмётся только канал 1024->100, и голова учит пуллинг по ним.
Размерность 2800 подобрана под Chronos (2816) и Toto (2560), чтобы контроль был сопоставим.

Ориентиры: наша свёртка 1,73808; CoLES на сырых рядах 1,737; ствол Toto 313m 1,76758 при
контроле 1,77794; ствол Chronos 1,77785 при контроле 1,77719.
"""
import sys, os, time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

logging.disable(logging.INFO)
sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from tsfm_public import PatchTSTFMForPrediction

MID = "ibm-granite/granite-timeseries-patchtst-fm-r1"
DEV = "cuda"
CTX, PRED = 320, 32
NPATCH, PDIM = 28, 100
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS_EMB = int(os.environ.get("BS_EMB", 256))
EPOCHS, BS, LR = 8, 512, 2e-3
NSAMP, GH, SIGMA = 96, 16, 0.8
NQ = 99
TAU = torch.arange(1, NQ + 1, dtype=torch.float32, device=DEV) / (NQ + 1)
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
CH = [G, V, C]

def win1(M, users, end):
    lo = end - CTX + 1
    if lo >= 0:
        return M[users, lo:end + 1].astype(np.float32)
    got = M[users, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), -lo), np.float32), got], 1)

print("загружаю PatchTST-FM ...", flush=True)
m = PatchTSTFMForPrediction.from_pretrained(MID).eval().to(DEV)
for p in m.parameters():
    p.requires_grad_(False)
print(f"  {sum(p.numel() for p in m.parameters())/1e6:.1f}М параметров, ствол заморожен", flush=True)
assert m.backbone.blocks[0].attn.qkv.weight.norm().item() > 50, "веса не загрузились"

BUF = {}
m.backbone.out_layer.register_forward_pre_hook(lambda mod, inp: BUF.__setitem__("h", inp[0].detach()))
gen = torch.Generator(device="cpu").manual_seed(0)
PROJ = (torch.randn(1024, PDIM, generator=gen) / np.sqrt(1024)).to(DEV)

def embed(anchor):
    outs = []
    for i in range(0, NU, BS_EMB):
        u = np.arange(i, min(i + BS_EMB, NU))
        x = torch.from_numpy(win1(G, u, anchor)).to(DEV)
        assert x.shape == (len(u), CTX), x.shape
        with torch.no_grad():
            m(past_values=x, prediction_length=PRED)
            h = BUF["h"]
            if i == 0:
                print(f"    представление: {tuple(h.shape)}", flush=True)
                assert h.shape[1] == NPATCH, h.shape
            outs.append((h @ PROJ).reshape(len(u), -1).half().cpu().numpy())
    return np.concatenate(outs, 0)

print("снимаю представления ...", flush=True)
t0 = time.time()
EMB = {}
for a in SEL_TR + [SEL_EV]:
    EMB[a] = embed(a)
    print(f"  якорь {a}: {EMB[a].shape}  {EMB[a].nbytes/1e9:.2f} ГБ  ({time.time()-t0:.0f}s)", flush=True)
D = EMB[SEL_EV].shape[1]
del m
torch.cuda.empty_cache()

def normalize(F):
    mu = np.mean([F[a].astype(np.float32).mean(0) for a in SEL_TR], 0)
    sd = np.maximum(np.mean([F[a].astype(np.float32).std(0) for a in SEL_TR], 0), 1e-3)
    for a in F:
        F[a] = np.clip((F[a].astype(np.float32) - mu) / sd, -8, 8).astype(np.float16)
    return F

EMB = normalize(EMB)

class Body(nn.Module):
    """общая часть: attention-пуллинг по патчам + последний патч"""
    def __init__(self, din, npatch, d=192):
        super().__init__()
        self.np_, self.pd = npatch, din // npatch
        self.q = nn.Parameter(torch.randn(1, 1, self.pd) * 0.02)
        self.pool = nn.MultiheadAttention(self.pd, 4, batch_first=True)
        self.body = nn.Sequential(nn.Linear(self.pd * 2, 512), nn.SiLU(),
                                  nn.Linear(512, d), nn.SiLU())
    def forward(self, e):
        B = e.shape[0]
        p = e.view(B, self.np_, self.pd)
        a = self.pool(self.q.expand(B, 1, -1), p, p, need_weights=False)[0].squeeze(1)
        return self.body(torch.cat([a, p[:, -1]], -1))

class GenHead(nn.Module):
    """наша генеративная голова: хазард + логнормаль по дням, frailty на выводе"""
    def __init__(self, din, npatch, d=192):
        super().__init__()
        self.b = Body(din, npatch, d)
        self.pos = nn.Parameter(torch.randn(HOR, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(d + 16, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, e):
        h = self.b(e); B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, HOR, -1),
                       self.pos.unsqueeze(0).expand(B, HOR, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

class QHead(nn.Module):
    """99 квантилей L=log1p(Y) месячной суммы; монотонность как у самой PatchTST-FM"""
    def __init__(self, din, npatch, d=192):
        super().__init__()
        self.b = Body(din, npatch, d)
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, NQ + 1))
    def forward(self, e):
        o = self.net(self.b(e))
        return o[:, :1] + torch.cumsum(Fn.softplus(o[:, 1:]) / NQ, 1)   # (B, 99), возрастает

def gen_loss(out, yd):
    hz, mu, ls = out
    pos = (yd > 0).float(); lyd = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyd - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def q_loss(q, yd):
    L = torch.log1p(yd.sum(1)).unsqueeze(1)           # (B, 1)
    d = L - q
    return torch.maximum(TAU * d, (TAU - 1) * d).mean(1).mean()

def gen_read(m_, feats):
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 1024):
            idx = np.arange(i, min(i + 1024, NU))
            hz, mu, ls = m_(torch.from_numpy(feats[idx].astype(np.float32)).to(DEV))
            hz = hz * scale; sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NSAMP, len(idx), HOR, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NSAMP, len(idx), HOR, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    return out

def q_read(m_, feats):
    """E[log1p Y] = интеграл log1p по подогнанной квантильной функции = средняя по квантилям"""
    out = np.empty(NU, np.float32); zf = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 1024):
            idx = np.arange(i, min(i + 1024, NU))
            q = m_(torch.from_numpy(feats[idx].astype(np.float32)).to(DEV)).clamp(min=0)
            out[idx] = q.mean(1).float().cpu().numpy()
            zf[idx] = (q < 1e-3).float().mean(1).cpu().numpy()
    return out, float(zf.mean())

def run(feats, kind, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m_ = (GenHead if kind == "gen" else QHead)(feats[SEL_EV].shape[1], NPATCH).to(DEV)
    lossf = gen_loss if kind == "gen" else q_loss
    opt = torch.optim.AdamW(m_.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    for _ in range(EPOCHS):
        m_.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64: continue
                e = torch.from_numpy(feats[a][idx].astype(np.float32)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                lossf(m_(e), yd).backward()
                nn.utils.clip_grad_norm_(m_.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    m_.eval()
    if kind == "gen":
        v, extra = gen_read(m_, feats[SEL_EV]), None
    else:
        v, extra = q_read(m_, feats[SEL_EV])
    del m_; torch.cuda.empty_cache()
    return v, extra

y = target(SEL_EV); ly = np.log1p(y)
print(f"\nдоля нулей в цели: {(y == 0).mean():.4f}  (столько нижних квантилей обязаны быть нулём)",
      flush=True)
ref = OUT / "h8_gauss.npy"
if ref.exists():
    r = np.load(ref); r = r - r.mean() + ly.mean()
    print(f"ориентиры: наша свёртка {rmsle(y, np.expm1(np.clip(r,0,13))):.5f}, "
          f"ствол Toto 313m 1.76758 (контроль 1.77794), Chronos 1.77785 (контроль 1.77719)",
          flush=True)

rs = np.random.default_rng(0)
W = (rs.standard_normal((len(CH) * CTX, D)) / np.sqrt(len(CH) * CTX)).astype(np.float32)
RND = {}
for a in SEL_TR + [SEL_EV]:
    X = np.log1p(np.maximum(np.concatenate([win1(M, np.arange(NU), a) for M in CH], 1), 0))
    RND[a] = (X @ W).astype(np.float16)
    del X
RND = normalize(RND)

print(f"\n{'ствол':>12} {'голова':>12} {'признаков':>10} {'RMSLE':>9} {'нулей':>7} {'сек':>7}",
      flush=True)
res = {}
for tname, feats in (("patchtst", EMB), ("randproj", RND)):
    for kind in ("gen", "q99"):
        t0 = time.time()
        v, zf = run(feats, "gen" if kind == "gen" else "q")
        sh = v - v.mean() + ly.mean()
        sc = rmsle(y, np.expm1(np.clip(sh, 0, 13)))
        res[f"{tname}_{kind}"] = v
        np.save(OUT / f"n3_{tname}_{kind}.npy", v)
        print(f"{tname:>12} {kind:>12} {D:10d} {sc:9.5f} "
              f"{('—' if zf is None else f'{zf:.3f}'):>7} {time.time()-t0:7.0f}", flush=True)

print()
for k in res:
    for k2 in res:
        if k < k2:
            print(f"corr({k}, {k2}) = {np.corrcoef(res[k], res[k2])[0,1]:.5f}")
if ref.exists():
    for k in res:
        print(f"corr({k}, наша дневная) = {np.corrcoef(res[k], np.load(ref))[0,1]:.5f}")
print("\ndone")
