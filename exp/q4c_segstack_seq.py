"""q4b closed the segment-weight idea on a reduced basis — LightGBM, CatBoost, an EB prior and
prev30. That basis has one honest weakness: it contains no sequence model, and the sequence
decorrelators are precisely the components whose value most plausibly varies with history
density. A dilated CNN needs a dense daily series to read anything; a zero-parameter frequency
prior works best exactly where that series is nearly empty. If segment-conditional weights are
real anywhere, it is on that pair.

So the same transfer test is repeated with a pure-sequence CNN added — the seq_pure geometry
from code/t23_seq_pure.py (no tabular head, since the head is what imports correlation from the
tree cluster).

Same four arms, same discipline: the segmented solution must beat BOTH the global weights and
the random-split null, and it must do so with a SHARED intercept, because the per-segment level
is independently known to be zero on the real test window (D_S = -0.0104 and -0.0025, both
within one SE of zero).
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L_DAY, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378),
         "HON": ([258, 230, 202, 174, 146], 288)}
NRAND = 60

def log16(n):
    return np.log1p(panel(n).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], 1)
CH = SEQ.shape[1]

def win(users, end):
    s = end - L_DAY + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class CNN(nn.Module):
    """seq_pure geometry: dilated CNN, no tabular head, two-output (gate, conditional mean)"""
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.head = nn.Sequential(nn.Linear(2 * h, 128), nn.BatchNorm1d(128), nn.SiLU(),
                                  nn.Linear(128, 2))
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))

def seq_fit_predict(tr, ev, seed=0):
    torch.manual_seed(seed)
    m = CNN(CH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(tr) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in tr}
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        m.train()
        for a in tr:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                y = torch.from_numpy(Y[a][idx]).to(DEV)
                o = m(x)
                pos = (y > 0).float()
                loss = (nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos,
                                                                       reduction="none")
                        + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            o = m(torch.from_numpy(win(idx, ev)).to(DEV))
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)

import lightgbm as lgb
from catboost import CatBoostRegressor

def components(tr, ev, X):
    Xtr = np.concatenate([X[a] for a in tr], 0)
    ytr = np.log1p(np.concatenate([target(a) for a in tr]))
    out = {}
    out["lgbm"] = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
        max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        n_jobs=11, verbose=-1, random_state=42, force_col_wise=True,
        objective="regression").fit(Xtr, ytr).predict(X[ev])
    out["catboost"] = CatBoostRegressor(
        iterations=700, learning_rate=0.09, depth=7, l2_leaf_reg=5.0, border_count=64,
        loss_function="RMSE", verbose=0, thread_count=11, random_seed=42,
        bootstrap_type="Bernoulli", subsample=0.8).fit(Xtr, ytr).predict(X[ev])
    del Xtr
    G = panel("gmv")
    B = np.stack([np.log1p(G[:, max(0, ev - 30 * k + 1):ev - 30 * (k - 1) + 1].sum(1))
                  for k in range(1, 7)], 1)
    mu, v = B.mean(1), B.var(1)
    gm, gv = mu.mean(), mu.var()
    w = gv / np.maximum(gv + v / 6.0, 1e-9)
    out["eb"] = w * mu + (1 - w) * gm
    out["seq_pure"] = seq_fit_predict(tr, ev)
    return out

def design(P, mask=None):
    cols = [P[k] for k in P]
    A = np.stack(cols + [np.ones(len(cols[0]))], 1)
    return A if mask is None else A[mask]

def fit_global(P, ly):
    return np.linalg.lstsq(design(P), ly, rcond=None)[0]

def fit_seg(P, ly, S, shared=False):
    if not shared:
        return (np.linalg.lstsq(design(P, S), ly[S], rcond=None)[0],
                np.linalg.lstsq(design(P, ~S), ly[~S], rcond=None)[0])
    base = np.stack([P[k] for k in P], 1)
    Z = np.concatenate([base * S[:, None], base * (~S)[:, None], np.ones((NU, 1))], 1)
    sol = np.linalg.lstsq(Z, ly, rcond=None)[0]
    n = base.shape[1]
    return np.r_[sol[:n], sol[-1]], np.r_[sol[n:2 * n], sol[-1]]

def apply_seg(P, wa, wb, S):
    A = design(P)
    return np.where(S, A @ wa, A @ wb)

CACHE = {}
for proto, (tr, ev) in PROTO.items():
    print(f"[{proto}] компоненты на {ds(ev)} (включая seq_pure) ...", flush=True)
    t0 = time.time()
    X, _ = build_features(sorted(set(tr + [ev])), verbose=False)
    P = components(tr, ev, X)
    bd = (panel("gmv")[:, :ev + 1] > 0).sum(1)
    y = target(ev)
    CACHE[proto] = (P, np.log1p(y), y, bd <= np.percentile(bd, 33))
    for k, v in P.items():
        print(f"    {k:9s} {rmsle(y, np.expm1(np.clip(v,0,13))):.5f}")
    print(f"    готово ({time.time()-t0:.0f}s)", flush=True)
    del X

rng = np.random.default_rng(0)
print(f"\n{'фит -> применение':22s} {'глобальные':>11} {'сегм.полные':>12} "
      f"{'сегм.общ.конст':>15} {'нуль(случ.)':>20}")
sf_d, ss_d = [], []
for src in PROTO:
    for dst in PROTO:
        if src == dst:
            continue
        Ps, lys, _, Ss = CACHE[src]
        Pd, lyd, yd, Sd = CACHE[dst]
        wg = fit_global(Ps, lys)
        g = rmsle(yd, np.expm1(np.clip(design(Pd) @ wg, 0, 13)))
        wa, wb = fit_seg(Ps, lys, Ss)
        sf = rmsle(yd, np.expm1(np.clip(apply_seg(Pd, wa, wb, Sd), 0, 13)))
        wa2, wb2 = fit_seg(Ps, lys, Ss, shared=True)
        ss = rmsle(yd, np.expm1(np.clip(apply_seg(Pd, wa2, wb2, Sd), 0, 13)))
        null = []
        for _ in range(NRAND):
            perm = rng.permutation(NU)
            qa, qb = fit_seg(Ps, lys, Ss[perm])
            null.append(rmsle(yd, np.expm1(np.clip(apply_seg(Pd, qa, qb, Sd[perm]), 0, 13))))
        null = np.array(null)
        sf_d.append(sf - g); ss_d.append(ss - g)
        print(f"{src+' -> '+dst:22s} {g:11.5f} {sf:12.5f} {ss:15.5f} "
              f"{null.mean():11.5f}±{null.std():.5f}")
print(f"\nсредняя дельта к глобальным: полные {np.mean(sf_d):+.5f}, "
      f"общая константа {np.mean(ss_d):+.5f}")
print(f"знаки (лучше/хуже): полные {sum(d<0 for d in sf_d)}/{sum(d>0 for d in sf_d)}, "
      f"общая константа {sum(d<0 for d in ss_d)}/{sum(d>0 for d in ss_d)}")
print("done")
