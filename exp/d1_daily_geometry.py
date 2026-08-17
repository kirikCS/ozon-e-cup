"""Three changes to the daily generative model, each attacking a different thing it cannot see.

The model earns weight +0.20 in the stack and is the only component reading the raw daily series
rather than window aggregates. Everything that recomposes the 165 aggregates is closed by
thirteen measurements, and the residual of the tree model is unpredictable (corr -0.00015), so
the remaining room is in representation, not in fitting. These are the three representation gaps
the current version has.

MULTIHORIZON. It is trained to predict exactly 30 days. Nothing stops it from predicting 7, 14,
30 and 60 at once from the same trunk — four times the supervision, and the 30-day head gets
regularised by heads that must agree with it on nested windows. The 7-day head in particular is
a much easier problem whose gradient sharpens the encoder early in training.

ACTIVE-DAY CLOCK. The window is 364 calendar days. For a client who visits twice a month that is
24 events padded with 340 zeros — the convolution spends its receptive field on emptiness.
evgru2 solved this by indexing on active days with explicit gaps and became the second-best
decorrelator in the stack, but it has no generative head. This arm puts the daily hazard head on
an active-day clock: the last 96 active days plus the gap length before each, so the model reads
"what happened" and "how long since" as separate signals.

CHANNEL REGIME. The catalog channel broke in April 2025 — its AOV jumped 37.42 -> 44.94 in one
month while search moved smoothly. The model eats catday/sord as two of eleven channels with no
way to know the regime changed. This arm adds an explicit regime covariate (days since the break,
clipped) and lets the channel embeddings interact with it.

Each arm is evaluated on SELECT against the current geometry, and its correlation with the
existing daily vector is reported — a variant that scores the same but decorrelates is worth
more here than one that scores better and duplicates.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, ds, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH = 364, 8, 512, 96, 16
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
HORIZONS = [7, 14, 30, 60]
BREAK_DAY = 90                      # 2025-04-01
NACT = 96                           # активных дней в варианте с событийными часами

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    P = panel(n).astype(np.float32)
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
ACT = (panel("visit") > 0)
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def cal_feats(a, hz=H):
    out = np.zeros((hz, 10), np.float32)
    for i in range(hz):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, d.weekday()] = 1.0
        out[i, 7] = d.day / 31.0
        out[i, 8] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        out[i, 9] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
    return out

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def win_active(users, end):
    """last NACT active days plus the gap before each — the event clock evgru2 uses"""
    out = np.zeros((len(users), CH + 1, NACT), np.float32)
    sub = ACT[users, :end + 1]
    for r, u in enumerate(users):
        idx = np.flatnonzero(sub[r])[-NACT:]
        if len(idx) == 0:
            continue
        vals = SEQ[u][:, idx].astype(np.float32)
        gaps = np.diff(np.concatenate([[idx[0]], idx])).astype(np.float32)
        out[r, :CH, -len(idx):] = vals
        out[r, CH, -len(idx):] = np.log1p(gaps)
    return out

class Enc(nn.Module):
    def __init__(self, c, h=128, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.h = h
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

class Model(nn.Module):
    def __init__(self, c, d=128, extra=0, multih=False):
        super().__init__()
        self.enc = Enc(c)
        self.multih = multih
        self.pos = nn.Parameter(torch.randn(max(HORIZONS), 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(self.enc.h * 3 + 10 + 16 + extra, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal, extra=None, hz=H):
        h = self.enc(x)
        B = h.shape[0]
        z = [h.unsqueeze(1).expand(B, hz, -1), cal.unsqueeze(0).expand(B, hz, -1),
             self.pos[:hz].unsqueeze(0).expand(B, hz, -1)]
        if extra is not None:
            z.append(extra.unsqueeze(1).expand(B, hz, -1))
        o = self.net(torch.cat(z, -1))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz_, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz_, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def regime(users, a):
    """days since the April-2025 catalog break, clipped and scaled"""
    return np.full((len(users), 1), np.clip((a - BREAK_DAY) / 365.0, -1, 2), np.float32)

def run(arm, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    act = arm == "active"
    extra = 1 if arm == "regime" else 0
    m = Model(CH + (1 if act else 0), extra=extra, multih=(arm == "multih")).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    hzs = HORIZONS if arm == "multih" else [H]
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {(a, hz): torch.from_numpy(cal_feats(a, hz)).to(DEV) for a in SEL_TR for hz in hzs}
    getw = win_active if act else win
    for ep in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(getw(idx, a)).to(DEV)
                ex = torch.from_numpy(regime(idx, a)).to(DEV) if extra else None
                # градиенты накапливаются ПО ОДНОМУ горизонту за раз. Сложение четырёх лоссов
                # в один граф держало в памяти четыре прямых прохода сразу и выдало OOM;
                # backward на каждом горизонте отдельно даёт тот же градиент вчетверо дешевле.
                opt.zero_grad(set_to_none=True)
                for hz in hzs:
                    yd = torch.from_numpy(G[idx, a + 1:a + 1 + hz].astype(np.float32)).to(DEV)
                    (loss_fn(*m(x, CAL[(a, hz)], ex, hz), yd)
                     * (1.0 if hz == H else 0.3)).backward()
                    del yd
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    cal = torch.from_numpy(cal_feats(SEL_EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    sigma = 0.8
    scale = float(np.sqrt(1 + np.pi * sigma ** 2 / 8))
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            ex = torch.from_numpy(regime(idx, SEL_EV)).to(DEV) if extra else None
            hzl, mu, ls = m(torch.from_numpy(getw(idx, SEL_EV)).to(DEV), cal, ex)
            hzl = hzl * scale
            sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hzl + float(zn) * sigma)
                u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NPATH, len(idx), H, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    del m
    torch.cuda.empty_cache()
    return out

y = target(SEL_EV)
ly = np.log1p(y)
ref = np.load(OUT / "q2c_daily_sel.npy") if (OUT / "q2c_daily_sel.npy").exists() else None
print(f"\n{'вариант':>10} {'RMSLE (уровень выровнен)':>25} {'corr с текущей':>16} {'сек':>7}")
if ref is not None:
    rs = ref - ref.mean() + ly.mean()
    print(f"{'текущая':>10} {rmsle(y, np.expm1(np.clip(rs,0,13))):25.5f} {1.0:16.5f} {0:7d}")
RES = {}
for arm in ("multih", "regime", "active"):
    t0 = time.time()
    try:
        v = run(arm)
    except Exception as e:
        # без явной очистки после исключения занятая память остаётся у процесса
        # и роняет следующую руку — на прошлом прогоне это чуть не случилось
        torch.cuda.empty_cache()
        print(f"{arm:>10}  ОШИБКА {type(e).__name__}: {str(e)[:70]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, ref)[0, 1] if ref is not None else float("nan")
    print(f"{arm:>10} {rmsle(y, np.expm1(np.clip(sh,0,13))):25.5f} {c:16.5f} "
          f"{time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"d1_{arm}_sel.npy", v)
if len(RES) > 1:
    print("\nвзаимные корреляции вариантов:")
    ks = list(RES)
    for a_ in ks:
        print(f"  {a_:>8} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:.4f}" for b in ks))
print("\ndone")
