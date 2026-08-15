"""Do the optimal stack weights actually differ between segments? Free local screen.

The saturated/cold idea, stated exactly: replace  sum_i w_i P_i  with
1_S * sum_i u_i P_i + 1_Sbar * sum_i v_i P_i. This is NOT what fx_segment_models tested
(those were per-segment BASE models, which lose cross-segment information and cost +0.0035)
and NOT what fx_shrink tested (a 2-parameter affine of ONE vector, which cannot change
which model dominates in a segment). It is the mixing that becomes conditional; the base
models stay pooled.

Before spending a leaderboard slot this is screened locally with the control discipline
from AGENT.md: the gain from the real split must beat the gain from RANDOM splits of the
same size. Any segmentation adds parameters and therefore always "wins" in-sample — the
null distribution is the whole point.

Measured on SELECT (eval 348) and confirmed on VAL (378), on four deliberately diverse
components: a tree, a boosted tree of another family, a zero-parameter frequency prior
(the natural cold pole) and a naive autoregression.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378)}
NRAND = 200

import lightgbm as lgb
from catboost import CatBoostRegressor

def components(tr, ev, X):
    """four diverse predictors of log1p(y) at the eval anchor"""
    Xtr = np.concatenate([X[a] for a in tr], 0)
    ytr = np.log1p(np.concatenate([target(a) for a in tr]))
    out = {}
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=63,
                          min_child_samples=100, max_bin=63, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=5.0, n_jobs=11, verbose=-1,
                          random_state=42, force_col_wise=True,
                          objective="regression").fit(Xtr, ytr)
    out["lgbm"] = m.predict(X[ev])
    m = CatBoostRegressor(iterations=700, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=11,
                          random_seed=42, bootstrap_type="Bernoulli",
                          subsample=0.8).fit(Xtr, ytr)
    out["catboost"] = m.predict(X[ev])
    del Xtr

    # zero-parameter frequency prior: empirical-Bayes shrunk mean of log1p over 30-day blocks
    G = panel("gmv")
    def eb(a):
        B = np.stack([np.log1p(G[:, max(0, a - 30 * k + 1):a - 30 * (k - 1) + 1].sum(1))
                      for k in range(1, 7)], 1)
        mu, v = B.mean(1), B.var(1)
        gm, gv = mu.mean(), mu.var()
        w = gv / np.maximum(gv + v / 6.0, 1e-9)
        return w * mu + (1 - w) * gm
    out["eb"] = eb(ev)
    out["prev30"] = np.log1p(G[:, ev - 29:ev + 1].sum(1))
    return out

def seg_weights(P, ly, mask):
    """OLS of the target on the components, restricted to a mask"""
    A = np.stack([P[k][mask] for k in P] + [np.ones(int(mask.sum()))], 1)
    return np.linalg.lstsq(A, ly[mask], rcond=None)[0]

def apply_w(P, w, mask):
    A = np.stack([P[k][mask] for k in P] + [np.ones(int(mask.sum()))], 1)
    return A @ w

for proto, (tr, ev) in PROTO.items():
    print(f"\n=== {proto}: обучение {tr} -> оценка {ev} ({ds(ev)}) ===")
    t0 = time.time()
    X, names = build_features(sorted(set(tr + [ev])), verbose=False)
    P = components(tr, ev, X)
    y = target(ev)
    ly = np.log1p(y)
    print(f"компоненты готовы ({time.time()-t0:.0f}s):")
    for k, v in P.items():
        print(f"    {k:9s} RMSLE {rmsle(y, np.expm1(np.clip(v,0,13))):.5f}")

    allm = np.ones(NU, bool)
    w_g = seg_weights(P, ly, allm)
    base = np.sqrt(np.mean((apply_w(P, w_g, allm) - ly) ** 2))
    print(f"\n  глобальный стек: {base:.6f}")
    print("  веса: " + ", ".join(f"{k} {w:+.3f}" for k, w in zip(list(P) + ["const"], w_g)))

    bd = (panel("gmv")[:, :ev + 1] > 0).sum(1)
    S = bd <= np.percentile(bd, 33)
    def gain_of(mask):
        wa, wb = seg_weights(P, ly, mask), seg_weights(P, ly, ~mask)
        pr = np.empty(NU)
        pr[mask] = apply_w(P, wa, mask)
        pr[~mask] = apply_w(P, wb, ~mask)
        return base - np.sqrt(np.mean((pr - ly) ** 2)), wa, wb

    g_real, wa, wb = gain_of(S)
    print(f"\n  сегмент 'холодные' (bd <= p33): доля {S.mean():.3f}")
    print("    веса холодных: " + ", ".join(f"{k} {w:+.3f}" for k, w in zip(list(P) + ["const"], wa)))
    print("    веса тёплых:   " + ", ".join(f"{k} {w:+.3f}" for k, w in zip(list(P) + ["const"], wb)))
    print(f"    выигрыш: {g_real:.6f}")

    rng = np.random.default_rng(0)
    null = np.array([gain_of(rng.permutation(S))[0] for _ in range(NRAND)])
    z = (g_real - null.mean()) / null.std()
    print(f"\n  КОНТРОЛЬ — {NRAND} случайных разметок той же мощности:")
    print(f"    нуль: среднее {null.mean():.6f}, sd {null.std():.6f}, максимум {null.max():.6f}")
    print(f"    реальная разметка: {g_real:.6f}  ->  z = {z:.2f}, "
          f"превзошла {100*(g_real>null).mean():.1f}% случайных")
    print(f"    вердикт: {'ЖИВО' if z > 3 else 'в пределах шума разметки'}")
print("\ndone")
