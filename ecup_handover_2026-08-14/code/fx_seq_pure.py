"""PURE sequence CNN — no tabular head. The decorrelation diagnostic.

The hybrid CNN correlates 0.9948 with the tree cluster, but that correlation is largely
imported by its tabular head (165 shared features). Nobody measured the pure-sequence
model's correlation. §6 math: at probe ~1.73 (corr(L) ~0.690) the admission bar is
g = corr(new, stack) <= 0.983 — plausible for a model that never sees the shared features.

2 seeds at SELECT geometry; prints probe scores and the corr matrix vs the cached
tabular ensembles and hybrid-CNN seeds.

Usage:  python -u code/fx_seq_pure.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, panel, target, rmsle, OUT

DEV = "cuda"
TR, EV = [318, 290, 262, 234], 348
L_DAY, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
SEEDS = [0, 1]
assert torch.cuda.is_available()

def log16(name):
    return np.log1p(panel(name).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], axis=1)
C = SEQ.shape[1]
Y = {a: target(a) for a in TR + [EV]}

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

class PureModel(nn.Module):
    def __init__(self, c, h=256, drop=0.15):
        super().__init__()
        self.enc = CNNEnc(c)
        self.body = nn.Sequential(nn.Linear(self.enc.out, h), nn.BatchNorm1d(h), nn.SiLU(),
                                  nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p = nn.Linear(128, 1); self.m = nn.Linear(128, 1)
    def forward(self, xs):
        z = self.body(self.enc(xs))
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()

def seq_slice(a, idx):
    s = a - L_DAY + 1
    return SEQ[idx, :, s:a + 1].astype(np.float32)

outs = []
for s in SEEDS:
    t0 = time.time()
    torch.manual_seed(s)
    model = PureModel(C).to(DEV)
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
                yb = torch.from_numpy(Y[a][idx].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xs), yb)
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
            o = model(torch.from_numpy(seq_slice(EV, idx)).to(DEV))
            acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
    lp = torch.cat(acc).numpy()
    outs.append(lp)
    np.save(OUT / f"fx_seq_pure_s{s}.npy", lp.astype(np.float32))
    print(f"  pure seed {s}: {rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(lp), 0, 13))):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()

ens = np.mean(outs, 0)
sc = rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(ens), 0, 13)))
print(f"pure 2-seed ens: {sc:.5f}", flush=True)

l2s = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
ens_2s = np.mean([np.load(OUT / f"fx_se_2s_s{s}.npy") for s in l2s], 0)
ens_l2 = np.mean([np.load(OUT / f"fx_se_l2_s{s}.npy") for s in l2s], 0)
hyb = np.mean([np.load(OUT / f"fx_seq_base_s{s}.npy") for s in range(3)], 0)
print(f"corr(pure, ens9_2s)  {np.corrcoef(ens, ens_2s)[0,1]:.5f}")
print(f"corr(pure, ens9_l2)  {np.corrcoef(ens, ens_l2)[0,1]:.5f}")
print(f"corr(pure, hybrid3)  {np.corrcoef(ens, hyb)[0,1]:.5f}")
print(f"corr(hybrid3, ens9_2s) {np.corrcoef(hyb, ens_2s)[0,1]:.5f}   <- reference")
ly = np.log1p(Y[EV])
print(f"corr(pure, target)   {np.corrcoef(ens, ly)[0,1]:.5f}")
mix = 0.5 * ens_2s + 0.5 * hyb
print(f"corr(pure, 2s+hyb mix) {np.corrcoef(ens, mix)[0,1]:.5f}  порог допуска "
      f"{np.corrcoef(ens, ly)[0,1]/0.70211:.4f}")

import os
os._exit(0)   # обход шумного teardown CUDA (ложные exit code 9)
