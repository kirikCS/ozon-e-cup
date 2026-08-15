"""Event-sequence GRU — the second NN family, different tokenization (diagnostic).

The daily-grid CNN family gave its orthogonal content once (t10: GRU/PatchTST on the
same grid correlate 0.998+ with it). This model tokenizes differently: a sequence of the
user's last 48 ACTIVE days (events), each with [log1p gmv, log1p searches, cart, ord,
log1p gap-to-prev, log1p days-to-anchor, dow sin, dow cos]. Gaps become explicit inputs
instead of implicit grid zeros — a genuinely different inductive bias.

SELECT diagnostic: 2 seeds, score + corr vs the span and vs seq_pure.
Kill: corr(event, seq_pure) >= 0.99 (sequence-family verdict extends) or score >= 1.80.

Usage:  python -u code/fx_event_gru.py
"""
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, panel, target, rmsle, OUT, D0

DEV = "cuda"
TR, EV = [318, 290, 262, 234], 348
K, EPOCHS, BS, LR = 48, 5, 2048, 1.5e-3
SEEDS = [0, 1]
assert torch.cuda.is_available()

G = panel("gmv")
S = panel("srch")
CA = panel("cart")
OR = panel("ord")
V = panel("visit")
WD0 = date(2025, 1, 1).weekday()


def build_events(a):
    """(NU, K, 8) float16, most recent event last; zero-padded on the left"""
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K)                      # last K active days
    ui, jj = np.nonzero(take)
    day = a - jj                                # actual day index
    slot = K - cc[ui, jj]                       # position in the sequence (recent -> K-1)
    E = np.zeros((NU, K, 8), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(OR[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    # gap to previous event: diff of day along each user's sequence
    prev = np.full(NU * K, -1, np.int64).reshape(NU, K)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    gap = np.zeros((NU, K), np.float32)
    valid = prev >= 0
    dshift = np.roll(d0, 1, axis=1)
    vshift = np.roll(valid, 1, axis=1)
    dshift[:, 0] = -1
    vshift[:, 0] = False
    g = np.where(valid & vshift, dshift - d0, 0)   # prev slot is EARLIER? check ordering
    # slot grows toward recent; prev slot = earlier event, so gap = d0 - dshift
    g = np.where(valid & vshift, d0 - dshift, 0)
    E[:, :, 4] = np.log1p(np.clip(g, 0, None)).astype(np.float16)
    return E


class EventGRU(nn.Module):
    def __init__(self, f=8, h=128):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)
        self.body = nn.Sequential(nn.Linear(2 * h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p = nn.Linear(128, 1)
        self.m = nn.Linear(128, 1)

    def forward(self, x):
        z = torch.nn.functional.silu(self.inp(x))
        o, _ = self.gru(z)
        pooled = torch.cat([o[:, -1], o.mean(1)], 1)
        z = self.body(pooled)
        return torch.cat([self.p(z), self.m(z)], 1)


def loss_fn(o, y):
    pos = (y > 0).float()
    ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()


print("building event tensors...", flush=True)
t0 = time.time()
EVT = {a: build_events(a) for a in TR + [EV]}
Y = {a: target(a) for a in TR + [EV]}
print(f"done ({time.time()-t0:.0f}s)", flush=True)

outs = []
for s in SEEDS:
    t0 = time.time()
    torch.manual_seed(s)
    model = EventGRU().to(DEV)
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
            o = model(torch.from_numpy(EVT[EV][idx].astype(np.float32)).to(DEV))
            acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
    lp = torch.cat(acc).numpy()
    outs.append(lp)
    np.save(OUT / f"fx_evgru_s{s}.npy", lp.astype(np.float32))
    print(f"  evgru seed {s}: {rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(lp), 0, 13))):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()

ens = np.mean(outs, 0)
sc = rmsle(Y[EV], np.expm1(np.clip(np.nan_to_num(ens), 0, 13)))
l2s = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
ens_2s = np.mean([np.load(OUT / f"fx_se_2s_s{s}.npy") for s in l2s], 0)
pure = np.mean([np.load(OUT / f"fx_seq_pure_s{s}.npy") for s in [0, 1]], 0)
hyb = np.mean([np.load(OUT / f"fx_seq_base_s{s}.npy") for s in range(3)], 0)
print(f"evgru 2-seed ens: {sc:.5f}")
print(f"corr(evgru, ens9_2s)   {np.corrcoef(ens, ens_2s)[0,1]:.5f}")
print(f"corr(evgru, seq_pure)  {np.corrcoef(ens, pure)[0,1]:.5f}")
print(f"corr(evgru, hybrid)    {np.corrcoef(ens, hyb)[0,1]:.5f}")
