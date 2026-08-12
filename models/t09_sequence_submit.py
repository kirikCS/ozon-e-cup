"""Sequence hybrid (dilated CNN over the raw daily panel + tabular features) at anchor 2026-02-13.

The only model in the set whose input is the day-by-day behaviour sequence rather than window
aggregates, so it is the last realistic source of a decorrelated error component. Three seeds are
averaged in log space: the stack only benefits if corr(L, new) clears ~0.684, so the model has to
be as strong as we can make it.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from common import H, NU, N_DAYS, panel, target, rmsle, Timer, build_features, OUT, CACHE, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294]
TR_PROBE = [348, 320, 292, 264]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
L_DAY, SEEDS, EPOCHS, BS, LR = 182, 3, 5, 2048, 1.5e-3
EL = 2.3284
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)
print("device:", DEV, "| anchors", ALL_A)

chans = [np.log1p(panel("gmv")).astype(np.float16),
         np.log1p(panel("gmv_search").astype(np.float32)).astype(np.float16),
         np.log1p(panel("ord").astype(np.float32)).astype(np.float16),
         np.log1p(panel("cart").astype(np.float32)).astype(np.float16),
         np.log1p(panel("srch").astype(np.float32)).astype(np.float16),
         panel("visit").astype(np.float16), panel("engaged").astype(np.float16)]
SEQ = np.stack(chans, axis=1); C = SEQ.shape[1]
print("SEQ", SEQ.shape, f"{SEQ.nbytes/1e9:.2f} GB")

X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
FDIM = X[PRED].shape[1]
Y = {a: target(a) for a in ALL_A if a != PRED}

def make_norm(anchors):
    Z = np.nan_to_num(np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in anchors], 0),
                      posinf=0, neginf=0)
    return Z.mean(0), np.maximum(Z.std(0), 1e-2)

def tab(a, mu, sd):
    z = np.nan_to_num(np.log1p(np.clip(X[a], 0, None)), posinf=0, neginf=0)
    return np.clip((z - mu) / sd, -8, 8).astype(np.float32)

def seq_slice(a, idx):
    s = a - L_DAY + 1
    if s >= 0:
        return SEQ[idx, :, s:a + 1].astype(np.float32)
    got = SEQ[idx, :, 0:a + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(idx), C, -s), np.float32), got], axis=2)

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

def run(tr, ev, seeds=SEEDS):
    mu, sd = make_norm(tr)
    TABS = {a: tab(a, mu, sd) for a in tr}
    YT = {a: Y[a].astype(np.float32) for a in tr}
    TE = tab(ev, mu, sd)
    outs = []
    for s in range(seeds):
        torch.manual_seed(s)
        model = Model(C, FDIM).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        steps = EPOCHS * len(tr) * (NU // BS)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
        rng = np.random.default_rng(s)
        for ep in range(EPOCHS):
            model.train()
            for a in tr:
                perm = rng.permutation(NU)
                for i in range(0, NU, BS):
                    idx = np.sort(perm[i:i + BS])
                    if len(idx) < 64: continue
                    xs = torch.from_numpy(seq_slice(a, idx)).to(DEV)
                    xt = torch.from_numpy(TABS[a][idx]).to(DEV)
                    yb = torch.from_numpy(YT[a][idx]).to(DEV)
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(xs, xt), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    if sched.last_epoch < steps - 1: sched.step()
        model.eval()
        acc = []
        with torch.no_grad():
            for i in range(0, NU, 8192):
                idx = np.arange(i, min(i + 8192, NU))
                xs = torch.from_numpy(seq_slice(ev, idx)).to(DEV)
                xt = torch.from_numpy(TE[idx]).to(DEV)
                o = model(xs, xt)
                acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
        outs.append(torch.cat(acc).numpy())
        print(f"    seed {s} done")
        del model; torch.cuda.empty_cache()
    return np.mean(outs, 0)

with Timer() as t:
    lp = run(TR_PROBE, FIT)
print(f"probe RMSLE at {ds(FIT)}: {rmsle(Y[FIT], np.expm1(np.clip(lp, 0, 13))):.5f}  ({t.dt:.0f}s)")

with Timer() as t:
    lf = run(TR_FINAL, PRED)
print(f"final done ({t.dt:.0f}s)")

lf = np.clip(np.nan_to_num(lf), 0, 13)
lf = lf - lf.mean() + EL                       # level already known from the LB parabola
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open(SUB / "seq_hybrid.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"seq_hybrid.csv: mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f} "
      f"p50={np.percentile(pred,50):.2f} p99={np.percentile(pred,99):.0f} max={pred.max():.0f}")
print("done")
