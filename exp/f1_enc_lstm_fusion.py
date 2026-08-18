"""Calendar encoder v2 and the event LSTM under one head, then bagged.

Two measured facts, combined:

  e1  the v2 encoder geometry — 364-day window, 14-day patches, CLS token, learned attention
      pooling — is worth -0.00767 over the old 182-day mean-pooled encoder. That was three
      quarters of the encoder's gap to the daily model, and all of it came from the INPUT: the
      generative head on top added +0.00047, i.e. nothing. So this build uses a scalar head,
      which is also eight times cheaper than simulating paths.

  e2  the event LSTM sits at corr 0.988 with the daily model against 0.996 for every calendar
      variant, and beat the GRU by 0.00084. Event time is the one axis on which all calendar
      readings are equally blind.

Why fuse rather than stack them separately: daily_active_bagged measured 1.67191 against a
threshold of 1.66774 — genuinely orthogonal (0.987) but too weak to pay its own way as a
component. Inside a fusion the same reading costs nothing, because the head uses it without the
stack having to price its accuracy. d5 already showed this works: the event clock alone scored
1.75449 and failed as a vector, yet added -0.00031 as a fourth trunk.

Three single-seed controls first — each trunk alone and the fusion — so the fusion's gain is
attributable. Then the fusion is bagged over four 80% client subsamples, the recipe that bought
-0.00252 on the leaderboard for the daily model, and both SELECT and FINAL vectors are written.
"""
import sys
import time
from datetime import date

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, PLEN = 364, 14
K_EV, H_EV = 96, 192
EPOCHS, BS, LR = 8, 512, 1.5e-3
NSEED, SUBSAMPLE = 4, 0.8
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
EL = 2.3284
WD0 = date(2025, 1, 1).weekday()

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G, S, CA, ORD_, V = (panel("gmv"), panel("srch"), panel("cart"), panel("ord"), panel("visit"))
SC, SO, SD, CD = panel("scart"), panel("sord"), panel("sday"), panel("catday")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def build_events(a):
    """last K_EV active days with explicit gaps — the evgru2 representation"""
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K_EV)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K_EV - cc[ui, jj]
    E = np.zeros((NU, K_EV, 12), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(ORD_[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    E[ui, slot, 8] = np.minimum(SC[ui, day], 20)
    E[ui, slot, 9] = np.minimum(SO[ui, day], 10)
    E[ui, slot, 10] = SD[ui, day]
    E[ui, slot, 11] = CD[ui, day]
    prev = np.full((NU, K_EV), -1, np.int64)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    valid = prev >= 0
    ds_ = np.roll(d0, 1, axis=1)
    vs = np.roll(valid, 1, axis=1)
    ds_[:, 0] = -1
    vs[:, 0] = False
    E[:, :, 4] = np.log1p(np.clip(np.where(valid & vs, d0 - ds_, 0), 0, None)).astype(np.float16)
    return E

class Block(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * d, d))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))

class EncTrunk(nn.Module):
    """v2 geometry: 364 days in 26 patches, CLS token, learned attention pooling"""
    def __init__(self, c, d=96, nl=3):
        super().__init__()
        self.n = L // PLEN
        self.emb = nn.Linear(c * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.out = d * 2
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, PLEN).permute(0, 2, 1, 3).reshape(B, self.n, CH * PLEN)
        z = torch.cat([self.cls.expand(B, 1, -1), self.emb(p) + self.pos], 1)
        for b in self.bl:
            z = b(z)
        pooled = self.pool(self.q.expand(B, 1, -1), z[:, 1:], z[:, 1:], need_weights=False)[0]
        return torch.cat([z[:, 0], pooled.squeeze(1)], -1)

class EvTrunk(nn.Module):
    def __init__(self, f=12, h=H_EV, layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.rnn = nn.LSTM(h, h, num_layers=layers, batch_first=True, dropout=0.1)
        self.out = 2 * h
    def forward(self, x):
        o, _ = self.rnn(Fn.silu(self.inp(x)))
        return torch.cat([o[:, -1], o.mean(1)], 1)

class Net(nn.Module):
    def __init__(self, arm, d=160):
        super().__init__()
        self.enc = EncTrunk(CH) if arm in ("enc", "fusion") else None
        self.ev = EvTrunk() if arm in ("lstm", "fusion") else None
        tot = (self.enc.out if self.enc else 0) + (self.ev.out if self.ev else 0)
        self.body = nn.Sequential(nn.Linear(tot, d), nn.BatchNorm1d(d), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(d, 128), nn.BatchNorm1d(128),
                                  nn.SiLU())
        self.p, self.m = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, xs, xe):
        parts = []
        if self.enc is not None:
            parts.append(self.enc(xs))
        if self.ev is not None:
            parts.append(self.ev(xe))
        z = self.body(torch.cat(parts, -1))
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def run(arm, TR, EV, EVT, seed=0, sub=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * sub), replace=False)) if sub < 1 else np.arange(NU)
    m = Net(arm).to(DEV)
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
                xs = torch.from_numpy(win(idx, a)).to(DEV) if m.enc is not None else None
                xe = (torch.from_numpy(EVT[a][idx].astype(np.float32)).to(DEV)
                      if m.ev is not None else None)
                yb = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(m(xs, xe), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 2048):
            idx = np.arange(i, min(i + 2048, NU))
            xs = torch.from_numpy(win(idx, EV)).to(DEV) if m.enc is not None else None
            xe = (torch.from_numpy(EVT[EV][idx].astype(np.float32)).to(DEV)
                  if m.ev is not None else None)
            o = m(xs, xe)
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out), npar

print("строю событийные тензоры ...", flush=True)
t0 = time.time()
EVT_S = {a: build_events(a) for a in SEL_TR + [SEL_EV]}
print(f"  {time.time()-t0:.0f}s", flush=True)

y = target(SEL_EV)
ly = np.log1p(y)
ref = OUT / "q2c_daily_sel.npy"
rv = np.load(ref) if ref.exists() else None
print(f"\n{'рука':>9} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>15} {'сек':>7}")
CTRL = {}
for arm in ("enc", "lstm", "fusion"):
    t0 = time.time()
    v, npar = run(arm, SEL_TR, SEL_EV, EVT_S)
    CTRL[arm] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, rv)[0, 1] if rv is not None else float("nan")
    print(f"{arm:>9} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:15.5f} {time.time()-t0:7.0f}", flush=True)
print(f"\nвыигрыш слияния над лучшим одиночным: "
      f"{min(rmsle(y, np.expm1(np.clip(CTRL[a]-CTRL[a].mean()+ly.mean(),0,13))) for a in ('enc','lstm')) - rmsle(y, np.expm1(np.clip(CTRL['fusion']-CTRL['fusion'].mean()+ly.mean(),0,13))):+.5f}")

print(f"\n=== бэггинг слияния: {NSEED} членов по {SUBSAMPLE:.0%} ===", flush=True)
acc = []
for s in range(NSEED):
    t0 = time.time()
    v, _ = run("fusion", SEL_TR, SEL_EV, EVT_S, seed=s, sub=SUBSAMPLE)
    acc.append(v)
    e = np.mean(acc, 0)
    e = e - e.mean() + ly.mean()
    print(f"  сид {s}: среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(e,0,13))):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
np.save(OUT / "f1_fusion_sel.npy", np.mean(acc, 0))
del EVT_S

print("\n=== FINAL ===", flush=True)
EVT_F = {a: build_events(a) for a in FIN_TR + [FIN_EV]}
accf = []
for s in range(NSEED):
    t0 = time.time()
    v, _ = run("fusion", FIN_TR, FIN_EV, EVT_F, seed=s, sub=SUBSAMPLE)
    accf.append(v)
    print(f"  сид {s} ({time.time()-t0:.0f}s)", flush=True)
E = np.mean(accf, 0)
f = E - E.mean() + EL
pl.DataFrame({"user_id": np.load("cache/uids.npy"),
              "predict": np.clip(np.expm1(f), 0, None)}).write_csv("submissions/enc_lstm_bagged.csv")
print(f"  submissions/enc_lstm_bagged.csv, mean log1p {f.mean():.4f}")
print("\ndone")
