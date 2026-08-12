"""Tier 4: neural networks on the tabular feature store (GPU).
MLP variants, the ZILN loss (Google), the exact two-head factorisation, MMoE and an MDME-style
multi-expert head (Kuaishou). Same two-protocol evaluation as Tier 3.
"""
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, record, Timer,
                    build_features, OUT)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 334, 320, 306, 292, 278, 264, 250]
TR_HON = [258, 244, 230, 216, 202, 188, 174, 160]
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))

X, NAMES = build_features(ALL_A)
try:
    BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
except Exception:
    BT = {}
for a in ALL_A:
    if a in BT:
        X[a] = np.concatenate([X[a], BT[a]], 1)
F = X[ALL_A[0]].shape[1]
print("feature dim", F)

Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)

# log1p + standardise, using train statistics only.
# The sd floor and the final clip matter: a couple of features are near-constant within a
# protocol, and dividing by their raw sd produced inputs of order 1e6 that blew the net up.
def prep(anchors, mu=None, sd=None):
    Z = np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in anchors], 0)
    Z = np.nan_to_num(Z, posinf=0, neginf=0)
    if mu is None:
        mu = Z.mean(0)
        sd = np.maximum(Z.std(0), 1e-2)
    return np.clip((Z - mu) / sd, -8, 8).astype(np.float32), mu, sd

PROTO = {"VAL": (TR_VAL, A, None), "HON": (TR_HON, AH, m_hon)}

# ------------------------------------------------------------------ losses
def ziln_loss(out, y):
    """zero-inflated lognormal negative log-likelihood (Wang, Liu, Miao 2019)"""
    logit, mu, logsig = out[:, 0], out[:, 1], out[:, 2]
    sig = torch.nn.functional.softplus(logsig) + 1e-4
    pos = (y > 0).float()
    cls = nn.functional.binary_cross_entropy_with_logits(logit, pos, reduction="none")
    safe = torch.clamp(y, min=1e-6)
    ln = 0.5 * ((torch.log(safe) - mu) / sig) ** 2 + torch.log(sig * safe * np.sqrt(2 * np.pi))
    return (cls + pos * ln).mean()

def ziln_predict_log(out):
    """RMSLE wants E[log1p(y)] -> p * E[log1p(y) | y>0]; approximate the latter by log1p(exp(mu))"""
    p = torch.sigmoid(out[:, 0])
    mu = out[:, 1]
    return p * torch.log1p(torch.exp(torch.clamp(mu, max=20)))

# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(self, f, h=(512, 256, 128), out=1, drop=0.15):
        super().__init__()
        L, d = [], f
        for k in h:
            L += [nn.Linear(d, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(drop)]
            d = k
        L += [nn.Linear(d, out)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)

class TwoHead(nn.Module):
    """exact RMSLE factorisation: E[log1p y] = P(y>0) * E[log1p y | y>0]"""
    def __init__(self, f, h=(512, 256, 128), drop=0.15):
        super().__init__()
        L, d = [], f
        for k in h:
            L += [nn.Linear(d, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(drop)]
            d = k
        self.body = nn.Sequential(*L)
        self.p = nn.Linear(d, 1)
        self.m = nn.Linear(d, 1)
    def forward(self, x):
        z = self.body(x)
        return torch.cat([self.p(z), self.m(z)], 1)

class MMoE(nn.Module):
    def __init__(self, f, n_exp=6, h=256, tasks=2, drop=0.15):
        super().__init__()
        self.exp = nn.ModuleList([nn.Sequential(
            nn.Linear(f, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.SiLU()) for _ in range(n_exp)])
        self.gate = nn.ModuleList([nn.Linear(f, n_exp) for _ in range(tasks)])
        self.tow = nn.ModuleList([nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1))
                                  for _ in range(tasks)])
    def forward(self, x):
        E = torch.stack([e(x) for e in self.exp], 1)            # B, n_exp, h
        outs = []
        for g, t in zip(self.gate, self.tow):
            w = torch.softmax(g(x), -1).unsqueeze(-1)
            outs.append(t((E * w).sum(1)))
        return torch.cat(outs, 1)

class MDME(nn.Module):
    """Kuaishou-style: split the target into ordered buckets, one expert per bucket,
    a gate over buckets, prediction = sum_k P(bucket k) * value_k"""
    def __init__(self, f, n_b=8, h=256, drop=0.15):
        super().__init__()
        self.n_b = n_b
        self.body = nn.Sequential(nn.Linear(f, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(512, h), nn.BatchNorm1d(h), nn.SiLU())
        self.gate = nn.Linear(h, n_b)
        self.exp = nn.ModuleList([nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1))
                                  for _ in range(n_b)])
    def forward(self, x):
        z = self.body(x)
        g = torch.softmax(self.gate(z), -1)
        v = torch.cat([e(z) for e in self.exp], 1)
        return torch.cat([g, v], 1)

# ------------------------------------------------------------------ training
def fit_predict(make_model, loss_fn, pred_fn, Xtr, ytr, Xev, epochs=12, bs=8192, lr=1e-3, wd=1e-5):
    n, f = Xtr.shape
    model = make_model(f).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs + 1))
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 32: continue
            xb = Xt[idx].to(DEV, non_blocking=True)
            yb = yt[idx].to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
    model.eval()
    outs = []
    with torch.no_grad():
        Xe = torch.from_numpy(Xev)
        for i in range(0, len(Xev), 32768):
            outs.append(pred_fn(model(Xe[i:i + 32768].to(DEV))).float().cpu())
    return torch.cat(outs).numpy()

CACHE_P = {}
def data(key):
    if key not in CACHE_P:
        tr, ev, mask = PROTO[key]
        Ztr, mu, sd = prep(tr)
        Zev, _, _ = prep([ev], mu, sd)
        ytr = np.concatenate([Y[a] for a in tr])
        CACHE_P[key] = (Ztr, ytr, Zev, Y[ev], mask)
    return CACHE_P[key]

def run(name, make_model, loss_fn, pred_fn, epochs=12, notes="", seeds=1):
    res, secs = {}, 0
    for key in PROTO:
        Ztr, ytr, Zev, yev, mask = data(key)
        with Timer() as t:
            ps = []
            for s in range(seeds):
                torch.manual_seed(s)
                ps.append(fit_predict(make_model, loss_fn, pred_fn, Ztr, ytr, Zev, epochs=epochs))
            logp = np.mean(ps, 0)
            # the largest observed target is ~54k, i.e. log1p ~ 10.9; anything above 13 is a
            # diverged head, not a prediction, so clamp before expm1 to avoid inf
            p = np.expm1(np.clip(np.nan_to_num(logp), 0, 13.0))
        secs += t.dt
        res[key] = rmsle(yev, p) if mask is None else rmsle(yev[mask], p[mask])
        if key == "VAL":
            np.save(OUT / f"pred_{name.replace('/', '_').replace(' ', '_')[:60]}.npy", p.astype(np.float32))
    return record(name, "neural", res["VAL"], res["HON"], secs, notes)

print("--- Tier 4: neural networks ---")

# plain MSE on log1p (prediction head already lives in log space)
run("MLP L2 on log1p", lambda f: MLP(f, out=1),
    lambda o, y: ((o[:, 0] - torch.log1p(y)) ** 2).mean(), lambda o: o[:, 0], epochs=12)
run("MLP L2 on log1p (3 seeds averaged)", lambda f: MLP(f, out=1),
    lambda o, y: ((o[:, 0] - torch.log1p(y)) ** 2).mean(), lambda o: o[:, 0], epochs=12, seeds=3)
run("MLP wide 1024-512-256 L2 on log1p", lambda f: MLP(f, h=(1024, 512, 256), out=1),
    lambda o, y: ((o[:, 0] - torch.log1p(y)) ** 2).mean(), lambda o: o[:, 0], epochs=16)

# ZILN
run("MLP + ZILN loss (Google)", lambda f: MLP(f, out=3), ziln_loss, ziln_predict_log, epochs=14,
    notes="zero-inflated lognormal NLL; prediction mapped to E[log1p]")

# exact two-head factorisation
def th_loss(o, y):
    pos = (y > 0).float()
    ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    mu = o[:, 1]
    reg = pos * (mu - ly) ** 2
    return (cls + reg).mean()
run("Two-head MLP: P(buy) x E[log1p | buy]", lambda f: TwoHead(f), th_loss,
    lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0), epochs=14,
    notes="exact RMSLE identity")
run("Two-head MLP (3 seeds averaged)", lambda f: TwoHead(f), th_loss,
    lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0), epochs=14, seeds=3)

# MMoE with the same two tasks
def mmoe_loss(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    reg = pos * (o[:, 1] - ly) ** 2
    return (cls + reg).mean()
run("MMoE (6 experts, 2 tasks)", lambda f: MMoE(f), mmoe_loss,
    lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0), epochs=14,
    notes="shared experts + per-task gates")

# MDME-style multi-expert over ordered buckets of log1p(y)
EDGES = torch.tensor([0.0, 1e-6, 2.5, 3.5, 4.3, 5.0, 5.8, 6.9], dtype=torch.float32)
def mdme_loss(o, y):
    nb = 8
    g, v = o[:, :nb], o[:, nb:]
    ly = torch.log1p(y)
    b = torch.bucketize(ly, EDGES.to(y.device)) - 1
    b = torch.clamp(b, 0, nb - 1)
    cls = nn.functional.cross_entropy(torch.log(g + 1e-8), b)
    val = ((v.gather(1, b.unsqueeze(1)).squeeze(1) - ly) ** 2).mean()
    return cls + val
def mdme_pred(o):
    nb = 8
    return (o[:, :nb] * torch.clamp(o[:, nb:], min=0)).sum(1)
run("MDME-style multi-expert (8 buckets)", lambda f: MDME(f), mdme_loss, mdme_pred, epochs=14,
    notes="divide-and-conquer over ordered target buckets, Kuaishou CIKM'22")

print("Tier 4 done")
