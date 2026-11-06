"""evgru3 at FINAL geometry — single batch of 6 seeds (the batch-drift lesson applied).

Config from the night diagnostic (2026-08-14): K=128 events, 12 features, h=256,
2-layer GRU + attention pooling, 8 epochs, BS 768. Local 1.75926 (vs evgru2 1.76388),
corr(evgru2) 0.9975 -> candidate to REPLACE evgru2's basis slot after a twink
measurement (family projection: LB ~1.656 vs evgru2's 1.6605).

Output: submissions/evgru3_final.csv (level EL), single-batch 6-seed log-average.
"""
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, N_DAYS, panel, target, OUT, CACHE

DEV = "cuda"
PRED = N_DAYS - 1
TR = [378, 350, 322, 294]
K, EPOCHS, BS, LR = 128, 8, 768, 1.0e-3
SEEDS = [0, 1, 2, 3, 4, 5]
EL = 2.3284
WD0 = date(2025, 1, 1).weekday()
assert torch.cuda.is_available()

G = panel("gmv"); S = panel("srch"); CA = panel("cart"); OR = panel("ord")
V = panel("visit"); SC = panel("scart"); SO = panel("sord")
SD = panel("sday"); CD = panel("catday")


def build_events(a):
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K - cc[ui, jj]
    E = np.zeros((NU, K, 12), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(OR[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    E[ui, slot, 8] = np.minimum(SC[ui, day], 20)
    E[ui, slot, 9] = np.minimum(SO[ui, day], 10)
    E[ui, slot, 10] = SD[ui, day]
    E[ui, slot, 11] = CD[ui, day]
    prev = np.full((NU, K), -1, np.int64)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    valid = prev >= 0
    dshift = np.roll(d0, 1, axis=1); vshift = np.roll(valid, 1, axis=1)
    dshift[:, 0] = -1; vshift[:, 0] = False
    g = np.where(valid & vshift, d0 - dshift, 0)
    E[:, :, 4] = np.log1p(np.clip(g, 0, None)).astype(np.float16)
    return E


class M(nn.Module):
    def __init__(self, f=12, h=256):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)
        self.att = nn.Linear(h, 1)
        self.body = nn.Sequential(nn.Linear(3 * h, 192), nn.BatchNorm1d(192), nn.SiLU())
        self.p = nn.Linear(192, 1); self.m = nn.Linear(192, 1)
    def forward(self, x):
        z = torch.nn.functional.silu(self.inp(x))
        o, _ = self.gru(z)
        w = torch.softmax(self.att(o).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = torch.cat([o[:, -1], o.mean(1), (o * w).sum(1)], 1)
        z = self.body(pooled)
        return torch.cat([self.p(z), self.m(z)], 1)


def loss_fn(o, y):
    pos = (y > 0).float()
    ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()


print("building event tensors...", flush=True)
t0 = time.time()
EVT = {a: build_events(a) for a in TR + [PRED]}
Y = {a: target(a) for a in TR}
print(f"done ({time.time()-t0:.0f}s)", flush=True)

outs = []
for s in SEEDS:
    t0 = time.time()
    torch.manual_seed(s)
    model = M().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (NU // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    rng = np.random.default_rng(s)
    for ep in range(EPOCHS):
        model.train()
        for a in TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = perm[i:i + BS]
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(EVT[a][idx].astype(np.float32)).to(DEV)
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
            o = model(torch.from_numpy(EVT[PRED][idx].astype(np.float32)).to(DEV))
            acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
    lp = torch.cat(acc).numpy()
    outs.append(lp)
    np.save(OUT / f"evgru3_final_s{s}.npy", lp.astype(np.float32))
    print(f"  final seed {s}: done ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()

ens = np.clip(np.nan_to_num(np.mean(outs, 0)), 0, 13)
ens = ens - ens.mean() + EL
pred = np.clip(np.expm1(ens), 0, None)
uids = np.load(CACHE / "uids.npy")
with open("submissions/evgru3_final.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"evgru3_final.csv written, mean log1p {np.log1p(pred).mean():.4f}")

import os
os._exit(0)
