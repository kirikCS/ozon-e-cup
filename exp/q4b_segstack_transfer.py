"""q4 said the segment-conditional weights are real (z = 20 on SELECT, z = 81 on VAL against
200 random splits of the same size). Two things must be checked before that means anything.

1. HOW MUCH OF IT IS JUST THE INTERCEPT. Looking at the fitted weights, cold and warm differ
   most in the constant (-0.085 vs -0.341) — that is a per-segment LEVEL, and the level was
   already probed directly on the real test window and came back zero (D_S = -0.0104, t=-0.96
   and -0.0025, t=-0.15). Only the part that survives a SHARED intercept is exploitable.

2. DOES IT TRANSFER. Every number in q4 is in-sample: the weights are fitted on the rows they
   are scored on, and the random control only prices the extra parameters, not the transfer.
   fx_shrink died exactly here — per-segment slopes existed at anchor 348 and did not carry to
   378. So: fit the segment weights at one anchor, apply them FROZEN at another, and compare
   against global weights fitted and frozen the same way.

Arms per anchor pair:
   global      one weight vector, fitted at A, applied at B
   seg_full    per-segment weights (own intercepts), fitted at A, applied at B
   seg_shared  per-segment slopes, ONE shared intercept, fitted at A, applied at B
   seg_random  same as seg_full on a random split — the transfer-time null
"""
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378),
         "HON": ([258, 230, 202, 174, 146], 288)}
NRAND = 60

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
    out["prev30"] = np.log1p(G[:, ev - 29:ev + 1].sum(1))
    return out

def design(P, mask=None):
    cols = [P[k] for k in P]
    A = np.stack(cols + [np.ones(len(cols[0]))], 1)
    return A if mask is None else A[mask]

def fit_global(P, ly):
    return np.linalg.lstsq(design(P), ly, rcond=None)[0]

def fit_seg(P, ly, S, shared_intercept=False):
    if not shared_intercept:
        return (np.linalg.lstsq(design(P, S), ly[S], rcond=None)[0],
                np.linalg.lstsq(design(P, ~S), ly[~S], rcond=None)[0])
    # общий свободный член: одна регрессия с сегментными наклонами и одной константой
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
    print(f"[{proto}] компоненты на {ds(ev)} ...", flush=True)
    t0 = time.time()
    X, _ = build_features(sorted(set(tr + [ev])), verbose=False)
    P = components(tr, ev, X)
    bd = (panel("gmv")[:, :ev + 1] > 0).sum(1)
    CACHE[proto] = (P, np.log1p(target(ev)), target(ev), bd <= np.percentile(bd, 33))
    print(f"    готово ({time.time()-t0:.0f}s)")
    del X

rng = np.random.default_rng(0)
print(f"\n{'фит -> применение':22s} {'глобальные':>11} {'сегм.полные':>12} "
      f"{'сегм.общ.конст':>15} {'нуль(случ.)':>18}")
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
        wa2, wb2 = fit_seg(Ps, lys, Ss, shared_intercept=True)
        ss = rmsle(yd, np.expm1(np.clip(apply_seg(Pd, wa2, wb2, Sd), 0, 13)))
        null = []
        for _ in range(NRAND):
            perm = rng.permutation(NU)
            Rs, Rd = Ss[perm], Sd[perm]
            qa, qb = fit_seg(Ps, lys, Rs)
            null.append(rmsle(yd, np.expm1(np.clip(apply_seg(Pd, qa, qb, Rd), 0, 13))))
        null = np.array(null)
        z = (null.mean() - sf) / null.std()
        print(f"{src+' -> '+dst:22s} {g:11.5f} {sf:12.5f} {ss:15.5f} "
              f"{null.mean():9.5f}±{null.std():.5f} z={z:+5.1f}")
print("\nчитать так: сегментные веса полезны, только если они бьют И глобальные, И нулевую разметку")
print("done")
