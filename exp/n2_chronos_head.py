"""Замороженный ствол Chronos-2 с нашей генеративной головой — параллель к k4 на Toto.

У Toto проверяли и zero-shot, и замороженный ствол со своей головой; у Chronos пока только
zero-shot (1.88316), который провалился ожидаемо — квантильный выход не имеет атома в нуле.

Здесь то же, что дало у Toto самый чистый ответ: ствол заморожен, сверху НАША генеративная
голова (хазард на нули, логнормаль на величину, frailty на зависимость дней) — то есть все три
элемента, которые по итогам m1/n1 оказались минимальным набором для смеси атома с непрерывной
частью. Вопрос ставится узко: несёт ли представление Chronos информацию о наших клиентах.

Отвод берётся сразу правильным, с учётом урока k3->k4: pipeline.embed отдаёт (переменная, 22
патча, 768), и патчи СОХРАНЯЮТСЯ — каждый сжимается фиксированной случайной проекцией 768->128
(иначе 6 якорей по 250k не влезают в 36 ГБ), а голова учит attention-пуллинг по ним и добирает
последний патч. У Toto именно расширение отвода учетверило отрыв от контроля.

Контроль обязателен и тот же: случайная линейная проекция сырого окна той же формы, через ту же
голову с тем же пуллингом. Без него нельзя отличить «представление полезно» от «голова неплоха».

Ориентиры: наша свёртка 1.73808, замороженный ствол Toto 313m 1.76758, Chronos zero-shot 1.88316.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from chronos import Chronos2Pipeline

DEV = "cuda"
CTX = int(os.environ.get("CTX", 320))
NPATCH, PDIM = 22, 128
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS_EMB = int(os.environ.get("BS_EMB", 256))
EPOCHS, BS, LR = 8, 512, 2e-3
NSAMP, GH, SIGMA = 96, 16, 0.8
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

print("загружаю chronos-2 ...", flush=True)
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV)
print(f"  {sum(p.numel() for p in pipe.model.parameters())/1e6:.1f}М параметров, ствол заморожен",
      flush=True)
gen = torch.Generator(device="cpu").manual_seed(0)
PROJ = (torch.randn(768, PDIM, generator=gen) / np.sqrt(768)).numpy().astype(np.float32)

def embed(anchor):
    outs = []
    for i in range(0, NU, BS_EMB):
        u = np.arange(i, min(i + BS_EMB, NU))
        e, _ = pipe.embed([win1(G, u, anchor)[k] for k in range(len(u))])
        arr = torch.stack([t[0] for t in e], 0).float().cpu().numpy()   # (b, 22, 768)
        if i == 0:
            print(f"    представление: {arr.shape}", flush=True)
        outs.append((arr @ PROJ).reshape(len(u), -1).astype(np.float16))
    return np.concatenate(outs, 0)

print("снимаю представления ...", flush=True)
t0 = time.time()
EMB = {}
for a in SEL_TR + [SEL_EV]:
    EMB[a] = embed(a)
    print(f"  якорь {a}: {EMB[a].shape}  {EMB[a].nbytes/1e9:.2f} ГБ  ({time.time()-t0:.0f}s)",
          flush=True)
D = EMB[SEL_EV].shape[1]
mu_ = np.mean([EMB[a].astype(np.float32).mean(0) for a in SEL_TR], 0)
sd_ = np.maximum(np.mean([EMB[a].astype(np.float32).std(0) for a in SEL_TR], 0), 1e-3)
for a in EMB:
    EMB[a] = np.clip((EMB[a].astype(np.float32) - mu_) / sd_, -8, 8).astype(np.float16)

class Head(nn.Module):
    """attention-пуллинг по патчам + последний патч, затем наша генеративная голова"""
    def __init__(self, din, npatch, d=192):
        super().__init__()
        self.np_, self.pd = npatch, din // npatch
        self.q = nn.Parameter(torch.randn(1, 1, self.pd) * 0.02)
        self.pool = nn.MultiheadAttention(self.pd, 4, batch_first=True)
        self.body = nn.Sequential(nn.Linear(self.pd * 2, 512), nn.SiLU(),
                                  nn.Linear(512, d), nn.SiLU())
        self.pos = nn.Parameter(torch.randn(HOR, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(d + 16, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, e):
        B = e.shape[0]
        p = e.view(B, self.np_, self.pd)
        a = self.pool(self.q.expand(B, 1, -1), p, p, need_weights=False)[0].squeeze(1)
        h = self.body(torch.cat([a, p[:, -1]], -1))
        z = torch.cat([h.unsqueeze(1).expand(B, HOR, -1),
                       self.pos.unsqueeze(0).expand(B, HOR, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); lyd = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyd - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(feats, npatch, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Head(feats[SEL_EV].shape[1], npatch).to(DEV)
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
                e = torch.from_numpy(feats[a][idx].astype(np.float32)).to(DEV)
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
            hz, mu, ls = m(torch.from_numpy(feats[SEL_EV][idx].astype(np.float32)).to(DEV))
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
    print(f"\nориентиры: наша свёртка {rmsle(y, np.expm1(np.clip(r,0,13))):.5f}, "
          f"ствол Toto 313m 1.76758, Chronos zero-shot 1.88316", flush=True)

print(f"\n{'рука':>14} {'признаков':>10} {'RMSLE':>9} {'сек':>7}", flush=True)
t0 = time.time(); v = run(EMB, NPATCH)
sh = v - v.mean() + ly.mean()
print(f"{'chronos_head':>14} {D:10d} {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
      f"{time.time()-t0:7.0f}", flush=True)
np.save(OUT / "n2_chronos_head.npy", v)

rs = np.random.default_rng(0)
W = (rs.standard_normal((len(CH) * CTX, D)) / np.sqrt(len(CH) * CTX)).astype(np.float32)
RND = {}
for a in SEL_TR + [SEL_EV]:
    X = np.log1p(np.maximum(np.concatenate([win1(M, np.arange(NU), a) for M in CH], 1), 0))
    RND[a] = (X @ W).astype(np.float16)
    del X
mu2 = np.mean([RND[a].astype(np.float32).mean(0) for a in SEL_TR], 0)
sd2 = np.maximum(np.mean([RND[a].astype(np.float32).std(0) for a in SEL_TR], 0), 1e-3)
for a in RND:
    RND[a] = np.clip((RND[a].astype(np.float32) - mu2) / sd2, -8, 8).astype(np.float16)
t0 = time.time(); v2 = run(RND, NPATCH)
sh2 = v2 - v2.mean() + ly.mean()
print(f"{'random_proj':>14} {D:10d} {rmsle(y, np.expm1(np.clip(sh2,0,13))):9.5f} "
      f"{time.time()-t0:7.0f}", flush=True)
print(f"\ncorr(chronos_head, random_proj) = {np.corrcoef(v, v2)[0,1]:.5f}")
if ref.exists():
    print(f"corr(chronos_head, наша дневная) = {np.corrcoef(v, np.load(ref))[0,1]:.5f}")
print("\ndone")
