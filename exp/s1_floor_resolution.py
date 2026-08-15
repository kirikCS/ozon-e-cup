"""Is the 1.6456 ceiling a property of the task, or of the grid it was measured on?

oracle_floor.py partitions clients into ~44,695 cells by five features at 20 quantile bins each,
hands every client the mean of their own cell, and bias-corrects the result to 1.6456. That
number is an estimate of E[Var(log1p y | cell)] at THAT conditioning resolution. It is not a
lower bound on what a model can do: finer conditioning splits the cells further and the estimate
falls. Whether it falls a little or a lot is an empirical question that was never asked.

Three independent estimators, because each fails differently:

  cells   pooled within-cell variance with ddof=1 — unbiased for E[Var(y|cell)] by construction,
          no correction factor needed. Swept over 3/5/8/12 features and 10/20/40 bins. If the
          estimate is still falling at the finest grid, it has not converged and the "ceiling"
          is just the last point of an unfinished sweep.

  kNN     conditional variance from k nearest neighbours in a standardised feature space,
          excluding the client itself. Handles many features without the cell-count explosion;
          biased upward for large k (over-smoothing) and noisy for small k, so the sweep over k
          brackets the truth rather than pinning it.

  resid   the direct question: is there structure left in the best local model's residual? Fit a
          fresh model to the residual, honestly out of sample. Signal there means the conditional
          mean has not been reached and no variance argument is needed at all.

An honest out-of-sample score is an UPPER bound on the floor (we can do at least that well);
these estimators approach it from below. The truth lies between, and how wide that interval is
decides whether chasing the fourth decimal is the right strategy at all.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
A = SEL_EV
X, names = build_features(sorted(set(SEL_TR + [A])), verbose=False)
y = target(A)
ly = np.log1p(y)
Xa = X[A]
print(f"якорь {ds(A)}, {NU} клиентов, {Xa.shape[1]} признаков")
print(f"Var(log1p y) = {ly.var():.4f}   RMSLE константы = {ly.std():.5f}")
print(f"опубликованный «потолок» = 1.64550 (corr 0.7039)\n")

# ранжирование признаков по одномерной корреляции с целью — тот же принцип, что в oracle_floor
r1 = np.array([abs(np.corrcoef(np.log1p(np.abs(Xa[:, j])) * np.sign(Xa[:, j]), ly)[0, 1])
               if Xa[:, j].std() > 1e-9 else 0.0 for j in range(Xa.shape[1])])
order = np.argsort(-np.nan_to_num(r1))
print("сильнейшие признаки: " + ", ".join(f"{names[j]} {r1[j]:.3f}" for j in order[:8]) + "\n")

def pooled_within(keys, v):
    """unbiased pooled within-cell variance: mean over cells of the ddof=1 cell variance,
    weighted by n-1. Cells of size 1 carry no information and are dropped, not zero-filled."""
    o = np.argsort(keys, kind="stable")
    ks, vs = keys[o], v[o]
    b = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    n = np.diff(b)
    csum = np.r_[0.0, np.cumsum(vs)]
    csq = np.r_[0.0, np.cumsum(vs ** 2)]
    s1 = csum[b[1:]] - csum[b[:-1]]
    s2 = csq[b[1:]] - csq[b[:-1]]
    ok = n >= 2
    ss = (s2 - s1 ** 2 / np.maximum(n, 1))[ok]
    return float(ss.sum() / (n[ok] - 1).sum()), int(len(n)), float(n.mean()), float(ok.mean())

print("=== 1. ячейки: как оценка зависит от разрешения ===")
print(f"{'призн.':>7} {'корзин':>7} {'ячеек':>9} {'ср.размер':>10} {'доля n>=2':>10} "
      f"{'оценка пола':>12}")
for J in (3, 5, 8, 12):
    for B in (10, 20, 40):
        cols = order[:J]
        q = np.stack([np.searchsorted(np.percentile(Xa[:, c], np.arange(100 / B, 100, 100 / B)),
                                      Xa[:, c]) for c in cols], 1).astype(np.int64)
        key = np.zeros(NU, np.int64)
        for j in range(J):
            key = key * (B + 1) + q[:, j]
        var, ncell, avg, frac = pooled_within(key, ly)
        print(f"{J:7d} {B:7d} {ncell:9d} {avg:10.2f} {frac:10.3f} {np.sqrt(var):12.5f}")

print("\n=== 2. k ближайших соседей (клиент исключён из своей оценки) ===")
from sklearn.neighbors import NearestNeighbors
D = 20
Z = Xa[:, order[:D]].astype(np.float64)
Z = np.log1p(np.abs(Z)) * np.sign(Z)
Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-6)
sub = np.random.default_rng(0).choice(NU, 60_000, replace=False)
t0 = time.time()
nn = NearestNeighbors(n_neighbors=51, algorithm="auto", n_jobs=11).fit(Z)
dist, ind = nn.kneighbors(Z[sub])
print(f"  соседи найдены за {time.time()-t0:.0f}s")
print(f"  {'k':>5} {'оценка пола':>12} {'RMSLE прогноза соседями':>26}")
for k in (5, 10, 25, 50):
    nb = ind[:, 1:k + 1]
    vals = ly[nb]
    var = float(np.mean(vals.var(1, ddof=1)))
    pred = vals.mean(1)
    print(f"  {k:5d} {np.sqrt(var):12.5f} {np.sqrt(np.mean((pred-ly[sub])**2)):26.5f}")

print("\n=== 3. осталась ли структура в остатке лучшей локальной модели ===")
import lightgbm as lgb
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
m = lgb.LGBMRegressor(**dict(LGB, random_state=42)).fit(Xtr, ytr)
p_ev = m.predict(Xa)
p_tr = m.predict(Xtr)
print(f"  модель: {rmsle(y, np.expm1(p_ev)):.5f}")
res_tr, res_ev = ytr - p_tr, ly - p_ev
m2 = lgb.LGBMRegressor(**dict(LGB, random_state=7)).fit(Xtr, res_tr)
q = m2.predict(Xa)
print(f"  предсказание остатка: corr с настоящим остатком {np.corrcoef(q, res_ev)[0,1]:+.5f}, "
      f"sd прогноза {q.std():.4f} против sd остатка {res_ev.std():.4f}")
for a_ in (0.0, 0.25, 0.5, 1.0):
    print(f"    модель + {a_:.2f}*предсказанный остаток: "
          f"{rmsle(y, np.expm1(np.clip(p_ev + a_*q, 0, 13))):.5f}")
print("\ndone")
