"""Does the search/catalogue channel split carry signal at the DAILY level for the CNN?

The seq hybrid is the only stack component that looks at the day-by-day sequence
(weight +0.423, the one architecture out of six that decorrelated). The base run feeds it
7 channels; this probe adds 4 channel-split funnel panels (search_to_cart, search_to_ord,
search-day flag, catalogue-day flag) and measures the paired per-seed delta.

Geometry is the SELECT protocol, shifted into sequence form:
  train anchors [318, 290, 262, 234] -> evaluate 348
(28-day spacing, exactly 30 days between last training anchor and evaluation, same as
t09's probe geometry; no target window overlaps the evaluation window.)

BTYD features are dropped from BOTH arms: btyd_feats.pkl lacks these anchors and its
warm-start makes recomputation order-dependent. The comparison stays paired and fair.

Usage:  python -u code/fx_seq_probe.py            # both arms, 3 seeds each
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import H, NU, N_DAYS, panel, target, rmsle, build_features, OUT, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TR, EV = [318, 290, 262, 234], 348
ALL_A = sorted(set(TR + [EV]))
L_DAY, SEEDS, EPOCHS, BS, LR = 182, 3, 5, 2048, 1.5e-3
assert torch.cuda.is_available(), "GPU required"

def log16(name):
    return np.log1p(panel(name).astype(np.float32)).astype(np.float16)

BASE_CH = [log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
           panel("visit").astype(np.float16), panel("engaged").astype(np.float16)]
EXT_CH = [log16("scart"), log16("sord"),
          panel("sday").astype(np.float16), panel("catday").astype(np.float16)]

X, NAMES = build_features(ALL_A)
FDIM = X[EV].shape[1]
Y = {a: target(a) for a in ALL_A}

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

def run_arm(SEQ, label):
    C = SEQ.shape[1]
    mu, sd = make_norm(TR)
    TABS = {a: tab(a, mu, sd) for a in TR}
    YT = {a: Y[a].astype(np.float32) for a in TR}
    TE = tab(EV, mu, sd)

    def seq_slice(a, idx):
        s = a - L_DAY + 1
        return SEQ[idx, :, s:a + 1].astype(np.float32)

    scores = []
    for s in range(SEEDS):
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
                xs = torch.from_numpy(seq_slice(EV, idx)).to(DEV)
                xt = torch.from_numpy(TE[idx]).to(DEV)
                o = model(xs, xt)
                acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
        lp = torch.cat(acc).numpy()
        p = np.expm1(np.clip(np.nan_to_num(lp), 0, 13))
        sc = rmsle(Y[EV], p)
        scores.append(sc)
        np.save(OUT / f"fx_seq_{label}_s{s}.npy", lp.astype(np.float32))
        print(f"  {label} seed {s}: RMSLE {sc:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    ens = np.mean([np.load(OUT / f"fx_seq_{label}_s{s}.npy") for s in range(SEEDS)], 0)
    sc_ens = rmsle(Y[EV], np.expm1(np.clip(ens, 0, 13)))
    print(f"  {label}: seeds {[round(x,5) for x in scores]}  mean {np.mean(scores):.5f}  "
          f"3-seed log-avg {sc_ens:.5f}", flush=True)
    return scores, sc_ens

print(f"device {DEV} | eval {ds(EV)} | tab dim {FDIM}")
SEQ_B = np.stack(BASE_CH, axis=1)
print(f"base arm: {SEQ_B.shape[1]} channels")
sb, eb = run_arm(SEQ_B, "base")
SEQ_E = np.concatenate([SEQ_B, np.stack(EXT_CH, axis=1)], axis=1)
del SEQ_B
print(f"ext arm: {SEQ_E.shape[1]} channels")
se, ee = run_arm(SEQ_E, "ext")

d = np.array(se) - np.array(sb)
print(f"\npaired deltas per seed: {[f'{x:+.5f}' for x in d]}")
print(f"mean single-seed delta {d.mean():+.5f} | ensemble delta {ee-eb:+.5f}")
