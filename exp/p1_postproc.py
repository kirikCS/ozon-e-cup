"""Post-processing families, screened with transfer and against the one control that matters.

Level and slope are already in production and already exact — the level from a three-point
parabola, the slope absorbed into the stack's weight sum. So the reference here is not the raw
prediction, it is the AFFINE-corrected prediction. Anything a family cannot beat affine by is
not post-processing, it is a rediscovery of the calibration we already have. r6 showed what
happens without that control: +0.015 of pure level-fitting looked like a finding.

Everything is fitted at one anchor and applied FROZEN at another. Isotonic calibration died
precisely because it was fitted on the January trough — the single seasonal dip of the year —
and carried "shrink towards January" onto a window containing Feb 23 and Mar 8.

Four families:

  affine    p' = a*p + c                          the control, already in production
  unc       p' = mu + (a + b*z)*(p - mu) + c      shrink harder where the estimate is less
                                                  certain, z = standardised disagreement
                                                  between the components. This is the first
                                                  time a per-client uncertainty exists at all:
                                                  fx_shrink had to proxy it with buyday deciles
                                                  and measured zero.
  bend      p' = a*p^gamma + c                    the natural next term after slope; gamma = 1
                                                  is exactly the affine control, so the family
                                                  strictly contains it and the gain is honest
  bins      p' = p + kappa * delta_j              per-prediction-bin bias, i.e. isotonic done
                                                  with shrinkage. kappa < 1 because bin biases
                                                  are noisy and the January lesson was about
                                                  trusting them fully
"""
import sys
import time

import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378),
         "HON": ([258, 230, 202, 174, 146], 288)}
NBIN = 12
G = panel("gmv")

def members(TR, EV, X):
    """diverse predictors; their spread is the per-client uncertainty signal"""
    Xtr = np.concatenate([X[a] for a in TR], 0)
    ytr = np.log1p(np.concatenate([target(a) for a in TR]))
    out = []
    for sd in (42, 101, 777):
        out.append(lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
            max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            n_jobs=11, verbose=-1, random_state=sd, force_col_wise=True,
            objective="regression").fit(Xtr, ytr).predict(X[EV]))
    out.append(CatBoostRegressor(iterations=700, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                                 border_count=64, loss_function="RMSE", verbose=0,
                                 thread_count=11, random_seed=42, bootstrap_type="Bernoulli",
                                 subsample=0.8).fit(Xtr, ytr).predict(X[EV]))
    del Xtr
    B = np.stack([np.log1p(G[:, max(0, EV - 30 * k + 1):EV - 30 * (k - 1) + 1].sum(1))
                  for k in range(1, 7)], 1)
    mu, v = B.mean(1), B.var(1)
    gm, gv = mu.mean(), mu.var()
    sh = gv / np.maximum(gv + v / 6.0, 1e-9)
    out.append(sh * mu + (1 - sh) * gm)
    P = np.stack(out)
    return P.mean(0), P.std(0)

def score(p, ly):
    return float(np.sqrt(np.mean((np.clip(p, 0, 13) - ly) ** 2)))

# ---- families: each returns a callable fitted on (p, z, ly) and applied to (p, z)
def fit_affine(p, z, ly):
    A = np.stack([p, np.ones_like(p)], 1)
    w = np.linalg.lstsq(A, ly, rcond=None)[0]
    return lambda q, s: w[0] * q + w[1], w

def fit_unc(p, z, ly):
    mu = p.mean()
    A = np.stack([p - mu, (p - mu) * z, np.ones_like(p)], 1)
    w = np.linalg.lstsq(A, ly - mu, rcond=None)[0]
    return lambda q, s: mu + w[0] * (q - mu) + w[1] * (q - mu) * s + w[2], w

def fit_bend(p, z, ly):
    q0 = np.maximum(p, 1e-6)
    def obj(t):
        a, g, c = t
        return np.mean((np.clip(a * q0 ** np.clip(g, 0.2, 3.0) + c, 0, 13) - ly) ** 2)
    r = minimize(obj, np.array([1.0, 1.0, 0.0]), method="Nelder-Mead",
                 options=dict(maxiter=600, xatol=1e-5, fatol=1e-9))
    a, g, c = r.x
    g = float(np.clip(g, 0.2, 3.0))
    return lambda q, s: a * np.maximum(q, 1e-6) ** g + c, np.array([a, g, c])

def fit_bins(p, z, ly, kappa=0.5):
    edges = np.percentile(p, np.linspace(100 / NBIN, 100 - 100 / NBIN, NBIN - 1))
    b = np.searchsorted(edges, p)
    d = np.array([np.mean(ly[b == j] - p[b == j]) if (b == j).sum() > 200 else 0.0
                  for j in range(NBIN)])
    def apply(q, s):
        bb = np.searchsorted(edges, q)
        return q + kappa * d[np.clip(bb, 0, NBIN - 1)]
    return apply, d

FAM = {"affine": fit_affine, "unc": fit_unc, "bend": fit_bend,
       "bins k=0.5": lambda p, z, ly: fit_bins(p, z, ly, 0.5),
       "bins k=1.0": lambda p, z, ly: fit_bins(p, z, ly, 1.0)}

CACHE = {}
for proto, (TR, EV) in PROTO.items():
    t0 = time.time()
    X, _ = build_features(sorted(set(TR + [EV])), verbose=False)
    p, s = members(TR, EV, X)
    ly = np.log1p(target(EV))
    z = (s - s.mean()) / max(s.std(), 1e-9)
    CACHE[proto] = (p, z, ly, s)
    print(f"[{proto}] {ds(EV)}: среднее по 5 моделям {score(p, ly):.5f}, "
          f"sd разброса {s.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    del X

print(f"\ncorr(разброс моделей, |ошибка|) — работает ли сигнал неопределённости вообще:")
for proto in PROTO:
    p, z, ly, s = CACHE[proto]
    print(f"  {proto:7s} {np.corrcoef(s, np.abs(p - ly))[0,1]:+.4f}")

print(f"\n=== перенос: подгонка на одном якоре, применение замороженным на другом ===")
print(f"{'фит -> прим.':20s} {'сырой':>9} {'affine':>9} " +
      " ".join(f"{k:>11}" for k in FAM if k != "affine") + "   сверх affine")
for src in PROTO:
    for dst in PROTO:
        if src == dst:
            continue
        ps, zs, lys, _ = CACHE[src]
        pd_, zd, lyd, _ = CACHE[dst]
        raw = score(pd_, lyd)
        res = {}
        for name, f in FAM.items():
            g, par = f(ps, zs, lys)
            res[name] = score(g(pd_, zd), lyd)
        aff = res["affine"]
        extra = " ".join(f"{aff-res[k]:+11.5f}" for k in FAM if k != "affine")
        print(f"{src+' -> '+dst:20s} {raw:9.5f} {aff:9.5f} " +
              " ".join(f"{res[k]:11.5f}" for k in FAM if k != "affine") + "  " + extra)
print("\nчитать так: колонки справа — выигрыш СВЕРХ аффинной поправки; всё, что меньше 0.0004,")
print("не отличимо от шума, а отрицательное значит, что семейство переобучается на якоре фита")
print("done")
