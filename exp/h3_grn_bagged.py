"""seq_hybrid geometry with the tabular block ROUTED instead of concatenated.

The stack holds two versions of the same CNN: seq_pure, which sees only the daily series and
correlates 0.9856 with the tree cluster, and seq_hybrid, which concatenates all 165 tabular
columns to the trunk output and correlates 0.9944. The hybrid is more accurate and less useful
per unit of accuracy, because concatenation hands the model the trees' own view of the client.

g4 measured the third option that nobody had tried: the same static block routed through a GRN
into a context vector instead of concatenated. On the daily generative model that was worth
-0.00114 over concatenation, well outside the +-0.0004 noise band, and g3 confirmed it
independently on raw columns (-0.00079). Neither test was on this geometry.

If the mechanism is what I claim — that routing keeps the static path separate rather than
letting it dominate the temporal one — then on the hybrid it should show up as lower correlation
at equal or better score. That is the arm worth having: the stack pays for orthogonality.

  pure     no tabular input at all               reproduces seq_pure
  concat   165 columns appended to the trunk     reproduces seq_hybrid
  grn      165 columns through a GRN as context  the untested third option

Scalar head throughout — this family has always used it, and e1 showed the generative head buys
nothing on non-convolutional readouts.
"""
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
NSEED, SUBSAMPLE, EL_LEVEL = 4, 0.8, 2.3284
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1

def log16(n):
    return np.log1p(panel(n).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], 1)
CH = SEQ.shape[1]
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

ANCH = sorted(set(SEL_TR + FIN_TR + [SEL_EV, FIN_EV]))
X, names = build_features(ANCH, verbose=False)
FS = len(names)
MU = np.mean([X[a].mean(0) for a in ANCH], 0)
SD = np.maximum(np.mean([X[a].std(0) for a in ANCH], 0), 1e-2)
ST = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in ANCH}
del X
print(f"  {FS} колонок", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class GRN(nn.Module):
    def __init__(self, din, dh, dout, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(din, dh)
        self.fc2 = nn.Linear(dh, dout * 2)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)
        self.dp = nn.Dropout(drop)
    def forward(self, x):
        h = self.dp(self.fc2(Fn.elu(self.fc1(x))))
        a, b = h.chunk(2, -1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))

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
    def __init__(self, arm, h=256, dctx=64, drop=0.15):
        super().__init__()
        self.arm = arm
        self.enc = CNNEnc(CH)
        self.grn = GRN(FS, 256, dctx) if arm == "grn" else None
        extra = FS if arm == "concat" else (dctx if arm == "grn" else 0)
        self.body = nn.Sequential(nn.Linear(self.enc.out + extra, h), nn.BatchNorm1d(h),
                                  nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p, self.m = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, xs, xt):
        z = self.enc(xs)
        if self.arm == "concat":
            z = torch.cat([z, xt], 1)
        elif self.arm == "grn":
            z = torch.cat([z, self.grn(xt)], 1)
        z = self.body(z)
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def run(arm, seed=0, TR=None, EV=None, sub=1.0):
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * sub), replace=False)) if sub < 1 else np.arange(NU)
    m = Model(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (len(keep) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in TR}
    for _ in range(EPOCHS):
        m.train()
        for a in TR:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(win(idx, a)).to(DEV)
                xt = torch.from_numpy(ST[a][idx]).to(DEV)
                yb = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(m(xs, xt), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            o = m(torch.from_numpy(win(idx, EV)).to(DEV),
                  torch.from_numpy(ST[EV][idx]).to(DEV))
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out), npar

y = target(SEL_EV)
ly = np.log1p(y)
best = "grn"
print(f"\n=== бэггинг руки {best} (ортогональная), SELECT и FINAL ===", flush=True)
print("  выбрана по ортогональности, не по скору: corr с pure 0.9574 против 0.9921 у concat", flush=True)
for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    acc = []
    for s_ in range(NSEED):
        t0 = time.time()
        v, _ = run(best, seed=s_, TR=TR, EV=EV, sub=SUBSAMPLE)
        acc.append(v)
        msg = ""
        if tag == "SELECT":
            e = np.mean(acc, 0); e = e - e.mean() + ly.mean()
            msg = f"  среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(e,0,13))):.5f}"
        print(f"  [{tag}] сид {s_}:{msg}  ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(acc, 0)
    if tag == "FINAL":
        f = E - E.mean() + EL_LEVEL
        pl.DataFrame({"user_id": np.load("cache/uids.npy"),
                      "predict": np.clip(np.expm1(f), 0, None)}).write_csv("submissions/hybrid_grn_bagged.csv")
        print(f"  submissions/hybrid_grn_bagged.csv, mean log1p {f.mean():.4f}")
    else:
        np.save(OUT / "h3_grn_bag_sel.npy", E)
print("\ndone")
