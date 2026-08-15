"""Channel decomposition of the target: gmv = gmv_search + gmv_cat, exactly.

Two ways to use an exact additive split, and they are not equally sound.

WRONG: predict log1p(Y_s) and log1p(Y_c) and add them. log1p(a+b) is not log1p a + log1p b,
so this is the Jensen trap again. Combining two channel predictions requires the joint
distribution of the two totals, not their log means.

SOUND, and what this tests:
  aux   one encoder, three heads — the total (used for the prediction) plus channel heads
        used ONLY as auxiliary supervision. Inference reads the total head, so no trap.
        The bet is on supervision density, the same bet as the daily model.
  mc    channel heads become distributional (mu, sigma per channel) and the total is
        assembled by Monte Carlo over the joint, taking log1p AFTER the sum. Correct, and
        it tests whether the split carries information the pooled head misses.

Scale check from the EDA: catalog is 7.4% of GMV mass and fires on 1.08% of rows, so the
total's log is dominated by search. The auxiliary variant is the one with a real prior;
the mc variant is the honest test of the decomposition itself.

Note on the apparatus: per-channel stacks cannot be calibrated the way the total is. The
leaderboard scores only the total, so no sub-target has a measurable R*. Channel models are
a way to BUILD a vector, never a way to calibrate one.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, PLEN = 182, 14
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
EPOCHS, BS = 6, 2048

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    try:
        P = panel(n).astype(np.float32)
    except Exception as e:
        print(f"  пропускаю {n}: {e}")
        continue
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G, GS = panel("gmv"), panel("gmv_search")
print(f"SEQ {SEQ.shape}")

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
F = X[SEL_EV].shape[1]
MU = np.concatenate([X[a] for a in SEL_TR], 0).mean(0)
SD = np.maximum(np.concatenate([X[a] for a in SEL_TR], 0).std(0), 1e-2)
Xs = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in anchors}

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def targets(users, a):
    tot = G[users, a + 1:a + 31].sum(1)
    src = GS[users, a + 1:a + 31].sum(1)
    return tot.astype(np.float32), src.astype(np.float32), (tot - src).astype(np.float32)

class Attn(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.nh, self.q, self.k, self.v, self.o = nh, *[nn.Linear(d, d) for _ in range(4)]
        self.dp = nn.Dropout(drop)
    def forward(self, x):
        B, T, D = x.shape
        h = self.nh
        q, k, v = [f(x).view(B, T, h, D // h).transpose(1, 2) for f in (self.q, self.k, self.v)]
        a = (q @ k.transpose(-1, -2)) / (D // h) ** 0.5
        z = (self.dp(a.softmax(-1)) @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(z)

class Block(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attn(d, nh, drop)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ff(self.n2(x))

class Net(nn.Module):
    """s-encoder over day patches plus a tabular tower; three two-output heads"""
    def __init__(self, fdim, d=96, nl=3, distr=False):
        super().__init__()
        self.emb = nn.Linear(CH * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, L // PLEN, d) * 0.02)
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.tab = nn.Sequential(nn.Linear(fdim, 256), nn.BatchNorm1d(256), nn.SiLU(),
                                 nn.Dropout(0.15), nn.Linear(256, d), nn.BatchNorm1d(d), nn.SiLU())
        nout = 3 if distr else 2
        self.h_tot = nn.Sequential(nn.Linear(2 * d, 128), nn.SiLU(), nn.Linear(128, nout))
        self.h_src = nn.Sequential(nn.Linear(2 * d, 128), nn.SiLU(), nn.Linear(128, nout))
        self.h_cat = nn.Sequential(nn.Linear(2 * d, 128), nn.SiLU(), nn.Linear(128, nout))
    def forward(self, xs, xt):
        B = xs.shape[0]
        n = L // PLEN
        p = xs.reshape(B, CH, n, PLEN).permute(0, 2, 1, 3).reshape(B, n, CH * PLEN)
        z = self.emb(p) + self.pos
        for b in self.bl:
            z = b(z)
        f = torch.cat([z.mean(1), self.tab(xt)], 1)
        return self.h_tot(f), self.h_src(f), self.h_cat(f)

def head_loss(o, y):
    pos = (y > 0).float()
    ly = torch.log1p(y)
    cls = Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()

def head_loss_d(o, y):
    pos = (y > 0).float()
    ly = torch.log1p(y)
    cls = Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    ls = o[:, 2].clamp(-4, 3)
    nll = 0.5 * (((ly - o[:, 1]) / ls.exp()) ** 2 + 2 * ls)
    return (cls + pos * nll).mean()

def run(tag, lam, distr, seed=0):
    torch.manual_seed(seed)
    m = Net(F, distr=distr).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=max(steps, 10))
    rng = np.random.default_rng(seed)
    lf = head_loss_d if distr else head_loss
    for ep in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(win(idx, a)).to(DEV)
                xt = torch.from_numpy(Xs[a][idx]).to(DEV)
                tt, ss, cc = [torch.from_numpy(v).to(DEV) for v in targets(idx, a)]
                ot, os_, oc = m(xs, xt)
                loss = lf(ot, tt) + lam * (lf(os_, ss) + lf(oc, cc))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    tot, mcv = [], []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            xs = torch.from_numpy(win(idx, SEL_EV)).to(DEV)
            xt = torch.from_numpy(Xs[SEL_EV][idx]).to(DEV)
            ot, os_, oc = m(xs, xt)
            tot.append((torch.sigmoid(ot[:, 0]) * ot[:, 1].clamp(min=0)).cpu().numpy())
            if distr:
                # сборка суммы по совместному распределению каналов, log1p ПОСЛЕ суммы
                acc = torch.zeros(len(idx), device=DEV)
                for _ in range(8):
                    s = torch.zeros(len(idx), device=DEV)
                    for o in (os_, oc):
                        u = (torch.rand(64, len(idx), device=DEV) < torch.sigmoid(o[:, 0]))
                        v = torch.expm1(o[:, 1] + o[:, 2].clamp(-4, 3).exp()
                                        * torch.randn(64, len(idx), device=DEV))
                        s = s + (u.float() * v.clamp(min=0)).mean(0)
                    acc += torch.log1p(s)
                mcv.append((acc / 8).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(tot), (np.concatenate(mcv) if distr else None)

y = target(SEL_EV)
print(f"\n=== канальная декомпозиция, оценка на {ds(SEL_EV)} ===")
res = {}
for tag, lam, distr in (("одна голова (контроль)", 0.0, False),
                        ("+каналы как вспомогательные, lam=0.3", 0.3, False),
                        ("+каналы, lam=1.0", 1.0, False),
                        ("распределения каналов + MC по сумме", 0.5, True)):
    t0 = time.time()
    p, mc = run(tag, lam, distr)
    res[tag] = p
    print(f"  {tag:38s} голова-итог {rmsle(y, np.expm1(np.clip(p,0,13))):.5f}  ({time.time()-t0:.0f}s)")
    if mc is not None:
        print(f"  {'':38s} сборка из каналов {rmsle(y, np.expm1(np.clip(mc,0,13))):.5f}")
        res["mc"] = mc
ks = list(res)
print("\nвзаимные корреляции вариантов:")
for a in ks:
    print(f"  {a[:34]:34s} " + " ".join(f"{np.corrcoef(res[a],res[b])[0,1]:.4f}" for b in ks))
print("\ndone")
