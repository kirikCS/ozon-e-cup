"""Three output parameterisations that differ from everything already in the stack.

Every model tried so far emits one number per client. These three change the shape of the output
itself, which is the remaining lever for decorrelation:

  Discrete-time hazard   30 daily buy probabilities; P(y>0) comes from 1 - prod(1 - h_d) rather
                         than from a single binary head. Uses the exact RMSLE identity
                         E[log1p y] = P(y>0) * E[log1p y | y>0].
  CC-OR-Net / RQ         cascaded ordinal decomposition over fixed quantile buckets with an
                         intra-bucket residual — ranking is guaranteed architecturally, and the
                         residual head is the depth-2 case of residual quantisation.
  PLE                    Progressive Layered Extraction: shared and task-specific experts split
                         level by level, the successor to MMoE (which scored 1.818 here).
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from zoo_common import load, emit, EL, TR_PROBE, TR_FINAL, FIT, PRED
from common import panel, target, rmsle, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
X, Y, NAMES, TOP = load()
G = panel("gmv")
H = 30

def daily_labels(a):
    """did the client buy on each of the 30 days after anchor a"""
    return (G[:, a + 1:a + H + 1] > 0).astype(np.float32)

# ------------------------------------------------------------------ 1. hazard
class Hazard(nn.Module):
    def __init__(self, f, h=(512, 256), drop=0.15):
        super().__init__()
        L, d = [], f
        for k in h:
            L += [nn.Linear(d, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(drop)]; d = k
        self.body = nn.Sequential(*L)
        self.haz = nn.Linear(d, H)          # 30 daily log-odds of buying
        self.mu = nn.Linear(d, 1)           # E[log1p(y) | y>0]
    def forward(self, x):
        z = self.body(x)
        return torch.cat([self.haz(z), self.mu(z)], 1)

def hazard_loss(o, y, dl):
    hz, mu = o[:, :H], o[:, H]
    pos = (y > 0).float(); ly = torch.log1p(y)
    bce = nn.functional.binary_cross_entropy_with_logits(hz, dl, reduction="none").mean(1)
    reg = pos * (mu - ly) ** 2
    return (bce + reg).mean()

def hazard_pred(o):
    hz, mu = o[:, :H], o[:, H]
    p_none = torch.exp(torch.log1p(-torch.sigmoid(hz).clamp(1e-6, 1 - 1e-6)).sum(1))
    return (1 - p_none) * torch.clamp(mu, min=0)

# ------------------------------------------------ 2. CC-OR-Net / residual quantisation
NB = 16
class CCOR(nn.Module):
    def __init__(self, f, h=(512, 256), drop=0.15):
        super().__init__()
        L, d = [], f
        for k in h:
            L += [nn.Linear(d, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(drop)]; d = k
        self.body = nn.Sequential(*L)
        self.cum = nn.Linear(d, NB - 1)     # P(L > edge_k), made monotone below
        self.res = nn.Linear(d, NB)         # intra-bucket residual
    def forward(self, x):
        z = self.body(x)
        return torch.cat([self.cum(z), self.res(z)], 1)

def make_ccor(edges, centers):
    ed = torch.tensor(edges, dtype=torch.float32, device=DEV)
    ce = torch.tensor(centers, dtype=torch.float32, device=DEV)
    def loss(o, y):
        cum, res = o[:, :NB - 1], o[:, NB - 1:]
        ly = torch.log1p(y)
        tgt = (ly.unsqueeze(1) > ed.unsqueeze(0)).float()       # cascaded ordinal targets
        bce = nn.functional.binary_cross_entropy_with_logits(cum, tgt, reduction="none").mean(1)
        b = torch.clamp(tgt.sum(1).long(), 0, NB - 1)
        r = res.gather(1, b.unsqueeze(1)).squeeze(1)
        return (bce + (r - (ly - ce[b])) ** 2).mean()
    def pred(o):
        cum, res = o[:, :NB - 1], o[:, NB - 1:]
        s = torch.sigmoid(cum)
        s, _ = torch.cummin(s, dim=1)                            # enforce monotone survival
        one = torch.ones_like(s[:, :1]); zero = torch.zeros_like(s[:, :1])
        surv = torch.cat([one, s, zero], 1)
        p = surv[:, :-1] - surv[:, 1:]                           # bucket probabilities
        return (p * (ce.unsqueeze(0) + res)).sum(1)
    return loss, pred

# ------------------------------------------------------------------ 3. PLE
class PLE(nn.Module):
    """two levels; at each level the shared experts feed both tasks while task-specific experts
    feed only their own, which is what separates PLE from MMoE"""
    def __init__(self, f, n_shared=3, n_task=2, h=256, drop=0.15, levels=2):
        super().__init__()
        self.levels = levels
        def blk(din):
            return nn.Sequential(nn.Linear(din, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop))
        self.sh, self.t0, self.t1, self.g0, self.g1, self.gs = (nn.ModuleList() for _ in range(6))
        din = f
        for lv in range(levels):
            self.sh.append(nn.ModuleList([blk(din) for _ in range(n_shared)]))
            self.t0.append(nn.ModuleList([blk(din) for _ in range(n_task)]))
            self.t1.append(nn.ModuleList([blk(din) for _ in range(n_task)]))
            self.g0.append(nn.Linear(din, n_shared + n_task))
            self.g1.append(nn.Linear(din, n_shared + n_task))
            self.gs.append(nn.Linear(din, n_shared + 2 * n_task))
            din = h
        self.head0 = nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1))
        self.head1 = nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x):
        a = b = s = x
        for lv in range(self.levels):
            S = torch.stack([e(s) for e in self.sh[lv]], 1)
            A = torch.stack([e(a) for e in self.t0[lv]], 1)
            B = torch.stack([e(b) for e in self.t1[lv]], 1)
            wa = torch.softmax(self.g0[lv](a), -1).unsqueeze(-1)
            wb = torch.softmax(self.g1[lv](b), -1).unsqueeze(-1)
            ws = torch.softmax(self.gs[lv](s), -1).unsqueeze(-1)
            na = (torch.cat([A, S], 1) * wa).sum(1)
            nb = (torch.cat([B, S], 1) * wb).sum(1)
            ns = (torch.cat([A, B, S], 1) * ws).sum(1)
            a, b, s = na, nb, ns
        return torch.cat([self.head0(a), self.head1(b)], 1)

def ple_loss(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()
ple_pred = lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)

# ------------------------------------------------------------------ trainer
def fit(make, loss_fn, pred_fn, anchors, ev, *, epochs=14, bs=8192, lr=1e-3, with_daily=False):
    Xtr = np.concatenate([X[a] for a in anchors], 0)
    ytr = np.concatenate([Y[a] for a in anchors]).astype(np.float32)
    DL = np.concatenate([daily_labels(a) for a in anchors], 0) if with_daily else None
    n, f = Xtr.shape
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    Dt = torch.from_numpy(DL) if with_daily else None
    torch.manual_seed(0)
    model = make(f).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * (n // bs + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10))
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 64: continue
            xb = Xt[idx].to(DEV); yb = yt[idx].to(DEV)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb, Dt[idx].to(DEV)) if with_daily else loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sch.last_epoch < steps - 1: sch.step()
    model.eval()
    acc = []
    with torch.no_grad():
        Xe = torch.from_numpy(X[ev])
        for i in range(0, len(X[ev]), 16384):
            acc.append(pred_fn(model(Xe[i:i + 16384].to(DEV))).float().cpu())
    del Xt, model
    torch.cuda.empty_cache()
    return torch.cat(acc).numpy()

# quantile edges for the ordinal cascade, taken from the probe-training targets
ly_all = np.log1p(np.concatenate([Y[a] for a in TR_PROBE]))
pos = ly_all[ly_all > 0]
EDGES = np.concatenate([[1e-6], np.quantile(pos, np.linspace(0, 1, NB)[1:-1])])
CENTERS = np.concatenate([[0.0], [(EDGES[i] + EDGES[i + 1]) / 2 for i in range(len(EDGES) - 1)],
                          [pos.max() * 0.9]])[:NB]
ccor_loss, ccor_pred = make_ccor(EDGES, CENTERS)

JOBS = [
    ("Discrete-time hazard (30 дней)", "zoo_hazard", lambda f: Hazard(f), hazard_loss, hazard_pred, True, 14),
    ("CC-OR-Net + residual quantisation", "zoo_ccor", lambda f: CCOR(f), ccor_loss, ccor_pred, False, 16),
    ("PLE (2 уровня, 2 задачи)", "zoo_ple", lambda f: PLE(f), ple_loss, ple_pred, False, 14),
]
print(f"=== специальные головы: {len(JOBS)} ===")
for name, fname, make, lf, pf, wd, ep in JOBS:
    print(f"\n{name}")
    try:
        lp = fit(make, lf, pf, TR_PROBE, FIT, epochs=ep, with_daily=wd)
        print(f"    probe {ds(FIT)}: {rmsle(Y[FIT], np.expm1(np.clip(np.nan_to_num(lp),0,13))):.5f}")
        emit(fit(make, lf, pf, TR_FINAL, PRED, epochs=ep, with_daily=wd), fname)
    except Exception as e:
        import traceback; print(f"    FAILED: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\ndone")
