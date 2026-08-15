"""Production pure-sequence CNN: the decorrelated component (diagnostic 2026-08-13).

No tabular head — the hybrid's 0.9944 correlation with the tree cluster is imported by
its 165 shared features; the pure model measured 0.9856 at equal score. Geometry and
training identical to t09/t14 minus the tabular input; 9 seeds, log-averaged, level EL.

Outputs: submissions/seq_pure_9seed.csv, model_out/t23_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, N_DAYS, panel, target, rmsle, OUT, CACHE, ds

DEV = "cuda"
PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294]
TR_PROBE = [348, 320, 292, 264]
L_DAY, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
SEEDS = list(range(9))
EL = 2.3284
assert torch.cuda.is_available()

def log16(name):
    return np.log1p(panel(name).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], axis=1)
C = SEQ.shape[1]
Y = {a: target(a) for a in sorted(set(TR_FINAL + TR_PROBE + [FIT]))}

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

def run(tr, ev, tag):
    outs = []
    for s in SEEDS:
        f = OUT / f"t23_{tag}_s{s}.npy"
        if f.exists():
            outs.append(np.load(f))
            print(f"  {tag} seed {s}: cached", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(s)
        model = PureModel(C).to(DEV)
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
                    if len(idx) < 64:
                        continue
                    xs = torch.from_numpy(
                        SEQ[idx, :, a - L_DAY + 1:a + 1].astype(np.float32)).to(DEV)
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
                o = model(torch.from_numpy(
                    SEQ[idx, :, ev - L_DAY + 1:ev + 1].astype(np.float32)).to(DEV))
                acc.append((torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)).float().cpu())
        lp = torch.cat(acc).numpy()
        np.save(f, lp.astype(np.float32))
        outs.append(lp)
        msg = f"  {tag} seed {s}: "
        if ev in Y:
            msg += f"{rmsle(Y[ev], np.expm1(np.clip(np.nan_to_num(lp), 0, 13))):.5f} "
        print(msg + f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return np.mean(outs, 0)

lp = run(TR_PROBE, FIT, "probe")
p9 = rmsle(Y[FIT], np.expm1(np.clip(np.nan_to_num(lp), 0, 13)))
print(f"probe ens9 at {ds(FIT)}: {p9:.5f}   (hybrid ens9 was 1.70542)", flush=True)
json.dump({"probe_ens9_pure": p9}, open(OUT / "t23_probe.json", "w"), indent=1)

lf = run(TR_FINAL, PRED, "final")
lf = np.clip(np.nan_to_num(lf), 0, 13)
lf = lf - lf.mean() + EL
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open(Path("submissions") / "seq_pure_9seed.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"seq_pure_9seed.csv: mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f}")

import os
os._exit(0)   # обход шумного teardown CUDA (ложные exit code 9)
