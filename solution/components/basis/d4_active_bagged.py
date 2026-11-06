"""Bagging applied where it has not been: the daily generative model.

The owner's idea — separate models on different client subsamples, blended — is closed for the
tree cluster by three independent measurements (7 seeds == 7 RFE subsets; a second pool with a
different seed beat the warm specialist; RF/ET land at g = 0.992-0.994). But the tree components
in the stack are already 9-seed averaged, and the daily generative model is not: it is ONE seed,
ONE simulation setting, and it carries weight +0.2147 — the least optimised component with the
largest weight after the blends themselves.

It also has a noise source the CNN does not, which is why the CNN's weak seed-averaging result
(9 seeds bought only -0.0001) does not transfer as a prediction here. Two sources, attacked
together:

  parameter noise    different seeds, and each member trained on a 80% client subsample —
                     the owner's construction exactly, applied to a model that never had it
  simulation noise   the Monte Carlo estimate of E[log1p sum]. Partial variance reduction
                     (16-node quadrature over the frailty) was already worth -0.0021 locally;
                     this run widens the quadrature to 24 nodes and doubles the paths

Members are averaged in the log scale, which is where the metric lives — averaging expm1 would
reintroduce the Jensen problem the whole model exists to avoid.

Produces both the SELECT vector (to see the local gain before spending a slot) and the FINAL
vector at anchor 408.
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
L = 364
EPOCHS, BS, NPATH, GH = 8, 1024, 192, 24
NSEED = 4
SUBSAMPLE = 0.8
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
NACT = 96
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
EL = 2.3284
OUTCSV = "submissions/daily_active_bagged.csv"

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    P = panel(n).astype(np.float32)
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}")

def cal_feats(a):
    out = np.zeros((H, 10), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, d.weekday()] = 1.0
        out[i, 7] = d.day / 31.0
        out[i, 8] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        out[i, 9] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
    return out

ACT = (panel("visit") > 0)

def win(users, end):
    """событийные часы: последние NACT активных дней плюс логарифм разрыва перед каждым.
    В d1 этот ствол дал corr 0.97739 с текущей моделью — самая низкая среди дневных
    вариантов, — при скоре 1.76165. Бэггинг, вытянувший календарную версию на -0.00252,
    к нему ещё не применяли."""
    out = np.zeros((len(users), CH + 1, NACT), np.float32)
    sub = ACT[users, :end + 1]
    for r, u in enumerate(users):
        idx = np.flatnonzero(sub[r])[-NACT:]
        if len(idx) == 0:
            continue
        out[r, :CH, -len(idx):] = SEQ[u][:, idx].astype(np.float32)
        out[r, CH, -len(idx):] = np.log1p(
            np.diff(np.concatenate([[idx[0]], idx])).astype(np.float32))
    return out

def _win_cal(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

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
    def __init__(self, c, d=128):
        super().__init__()
        self.enc = Enc(c)
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(self.enc.h * 3 + 10 + 16, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal):
        h = self.enc(x)
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                       cal.unsqueeze(0).expand(B, H, -1),
                       self.pos.unsqueeze(0).expand(B, H, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def fit(anchors, seed):
    """each member sees a different 80% of clients — the subsample bagging"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * SUBSAMPLE), replace=False))
    m = Model(CH + 1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(anchors) * (len(keep) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in anchors}
    for _ in range(EPOCHS):
        m.train()
        for a in anchors:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a]), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    return m.eval()

@torch.no_grad()
def simulate(m, anchor, sigma, npath=NPATH, chunk=256, users=None):
    cal = torch.from_numpy(cal_feats(anchor)).to(DEV)
    users = np.arange(NU) if users is None else users
    out = np.empty(len(users), np.float32)
    p0 = np.empty(len(users), np.float32)
    scale = float(np.sqrt(1 + np.pi * sigma ** 2 / 8))
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    for i in range(0, len(users), chunk):
        idx = users[i:i + chunk]
        hz, mu, ls = m(torch.from_numpy(win(idx, anchor)).to(DEV), cal)
        hz = hz * scale
        sd = ls.exp()
        b = len(idx)
        acc = torch.zeros(b, device=DEV)
        acc0 = torch.zeros(b, device=DEV)
        for zn, wn in zip(nodes, wts):
            p = torch.sigmoid(hz + float(zn) * sigma)
            u = torch.rand(npath, b, H, device=DEV) < p.unsqueeze(0)
            v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                            * torch.randn(npath, b, H, device=DEV))
            tot = (u.float() * v.clamp(min=0)).sum(-1)
            acc += float(wn) * torch.log1p(tot).mean(0)
            acc0 += float(wn) * (tot > 0).float().mean(0)
        out[i:i + len(idx)] = acc.float().cpu().numpy()
        p0[i:i + len(idx)] = acc0.float().cpu().numpy()
    return out, p0

def fit_sigma(m, anchors):
    sub = np.arange(0, NU, 12)
    emp = np.mean([np.mean(target(a)[sub] > 0) for a in anchors])
    best = None
    for s in (0.4, 0.6, 0.8, 1.0, 1.2):
        got = np.mean([simulate(m, a, s, npath=24, users=sub)[1].mean() for a in anchors])
        if best is None or abs(got - emp) < best[0]:
            best = (abs(got - emp), s)
    return best[1]

for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    print(f"\n=== {tag}: {NSEED} членов, подвыборка {SUBSAMPLE:.0%} клиентов ===", flush=True)
    acc = []
    for s in range(NSEED):
        t0 = time.time()
        m = fit(TR, seed=s)
        sig = fit_sigma(m, TR)
        v, _ = simulate(m, EV, sig)
        acc.append(v)
        del m
        torch.cuda.empty_cache()
        msg = ""
        if tag == "SELECT":
            y = target(EV)
            sh = v - v.mean() + np.log1p(y).mean()
            cur = np.mean(acc, 0)
            cs = cur - cur.mean() + np.log1p(y).mean()
            msg = (f"  член {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}  "
                   f"среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(cs,0,13))):.5f}")
        print(f"  сид {s}: sigma {sig}{msg}  ({time.time()-t0:.0f}s)", flush=True)
    ens = np.mean(acc, 0)
    if tag == "SELECT":
        np.save(OUT / "c5_active_sel.npy", ens)
    else:
        import polars as pl
        f = ens - ens.mean() + EL
        uid = np.load("cache/uids.npy")
        pl.DataFrame({"user_id": uid,
                      "predict": np.clip(np.expm1(f), 0, None)}).write_csv(OUTCSV)
        print(f"  {OUTCSV}, mean log1p {f.mean():.4f}")
print("\ndone")
