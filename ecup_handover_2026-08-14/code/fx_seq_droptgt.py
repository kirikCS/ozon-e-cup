"""Does dropping the calendar columns help the CNN at the LB-like anchor?

The tabular head takes the full store including tgt_month/tgt_doy/hist_days; at eval
month outside the training range the MLP extrapolates them too (linearly, so milder than
trees — measured here). Probe geometry = t09's: train [348, 320, 292, 264] -> eval 378,
seeds 0-2; the with-calendar reference from t14 on this pipeline is ens3 = 1.70869.

Usage:  python -u code/fx_seq_droptgt.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, panel, target, rmsle, build_features, OUT, ds

DEV = "cuda"
TR, EV = [348, 320, 292, 264], 378
ALL_A = sorted(set(TR + [EV]))
L_DAY, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
SEEDS = [0, 1, 2]
DROP = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}
REF_ENS3 = 1.70869
assert torch.cuda.is_available()

def log16(name):
    return np.log1p(panel(name).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], axis=1)
C = SEQ.shape[1]
X, NAMES = build_features(ALL_A)
keep = [i for i, n in enumerate(NAMES) if n not in DROP]
for a in ALL_A:
    X[a] = X[a][:, keep]
FDIM = X[EV].shape[1]
Y = {a: target(a) for a in ALL_A}
print(f"tab dim {FDIM} (dropped {len(DROP)})")

def make_norm(anchors):
    Z = np.nan_to_num(np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in anchors], 0),
                      posinf=0, neginf=0)
    return Z.mean(0), np.maximum(Z.std(0), 1e-2)

def tab(a, mu, sd):
    z = np.nan_to_num(np.log1p(np.clip(X[a], 0, None)), posinf=0, neginf=0)
    return np.clip((z - mu) / sd, -8, 8).astype(np.float32)

class CNNEnc(nn.Module):
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.out = h * 2
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z.max(-1).values], -1)

class Model(nn.Module):
    def __init__(self, c, d_tab, h=256, drop=0.15):
        super().__init__()
        self.enc = CNNEnc(c)
        d = self.enc.out + d_tab
        self.body = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p = nn.Linear(128, 1); self.m = nn.Linear(128, 1)
    def forward(self, xs, xt):
        z = self.body(torch.cat([self.enc(xs), xt], 1))
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()

def seq_slice(a, idx):
    s = a - L_DAY + 1
    return SEQ[idx, :, s:a + 1].astype(np.float32)

mu, sd = make_norm(TR)
TABS = {a: tab(a, mu, sd) for a in TR}
YT = {a: Y[a].astype(np.float32) for a in TR}
TE = tab(EV, mu, sd)
outs = []
for s in SEEDS:
    t0 = time.time()
    torch.manual_seed(s)
    model = Model(C, FDIM).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (NU // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    rng = np.random.default_rng(s)
    for ep in range(EPOCHS):
        model.train()
        for a in TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(seq_slice(a, idx)).to(DEV)
                xt = torch.from_numpy(TABS[a][idx]).to(DEV)
                yb = torch.from_numpy(YT[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xs, xt), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if sched.last_epoch < steps - 1:
                    sched.step()
    model.eval()
    acc = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            o = model(torch.from_numpy(seq_slice(EV, idx)).to(DEV),
                      torch.from_numpy(TE[idx]).to(DEV))
            acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
    lp = torch.cat(acc).numpy()
    outs.append(lp)
    print(f"  droptgt seed {s}: {rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(lp), 0, 13))):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
ens = np.mean(outs, 0)
sc = rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(ens), 0, 13)))
print(f"droptgt ens3 {sc:.5f}  vs with-calendar ens3 {REF_ENS3}  delta {sc-REF_ENS3:+.5f}")
