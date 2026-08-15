"""Owner's request (2026-08-13): split clients into types (active/passive), try removing
some from training — segment-specialist models and training-row filtering.

Distinct from what's closed: fx_shrink killed per-segment OUTPUT calibration; this tests
per-segment MODELS (each specialises its splits) and row filtering. Arms on SELECT:

  base          one model, all rows (reference: 1.74242/1.74201/1.74209 @ colsample 1.0)
  split_gmv30   two models: rows with gmv_30>0 vs ==0 at their anchor; eval routed by segment
  split_visit30 same split by visit_30
  drop_dead     one model, training rows with visit_180==0 removed (deadweight)
  winsor995     one model, log-target clipped at its p99.5 (outlier removal — "кого-то убрать")
  upweight_act  one model, active (gmv_30>0) rows weighted 2x

3 seeds each; noise band of this screen ~±0.0004.

Usage:  python -u code/fx_segment_models.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, target, rmsle, build_features, NU

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = [42, 101, 777]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
G = panel("gmv")
V = panel("visit")


def wsum(M, a, w):
    return M[:, max(0, a - w + 1):a + 1].sum(axis=1)


seg = {}
for a in anchors:
    seg[a] = {
        "gmv30": wsum(G, a, 30) > 0,
        "visit30": wsum(V, a, 30) > 0,
        "visit180": wsum(V, a, 180) > 0,
    }

lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
yev = target(SEL_EV)
Xev = X[SEL_EV]
tr_seg = {k: np.concatenate([seg[a][k] for a in SEL_TR]) for k in ["gmv30", "visit30", "visit180"]}
ev_seg = {k: seg[SEL_EV][k] for k in ["gmv30", "visit30"]}
for k, v in tr_seg.items():
    print(f"train share {k}: {v.mean():.3f}", flush=True)


def fit_predict(Xt, yt, Xe, seed, sw=None):
    m = lgb.LGBMRegressor(**LGB, random_state=seed)
    m.fit(Xt, yt, sample_weight=sw)
    return m.predict(Xe)


base_scores = {}
for seed in SEEDS:
    t0 = time.time()
    lp = fit_predict(Xtr, lytr, Xev, seed)
    base_scores[seed] = rmsle(yev, np.expm1(np.clip(lp, 0, None)))
    print(f"  base         seed {seed:>4}: {base_scores[seed]:.5f} ({time.time()-t0:.0f}s)",
          flush=True)


def arm(label, fn):
    deltas = []
    for seed in SEEDS:
        t0 = time.time()
        lp = fn(seed)
        s = rmsle(yev, np.expm1(np.clip(lp, 0, None)))
        deltas.append(s - base_scores[seed])
        print(f"  {label:<12} seed {seed:>4}: {s:.5f}  d {deltas[-1]:+.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    d = np.array(deltas)
    print(f"  {label}: mean delta {d.mean():+.5f}  spread [{d.min():+.5f} .. {d.max():+.5f}]",
          flush=True)


def split_arm(key):
    def fn(seed):
        m_tr, m_ev = tr_seg[key], ev_seg[key]
        lp = np.empty(NU)
        lp[m_ev] = fit_predict(Xtr[m_tr], lytr[m_tr], Xev[m_ev], seed)
        lp[~m_ev] = fit_predict(Xtr[~m_tr], lytr[~m_tr], Xev[~m_ev], seed)
        return lp
    return fn


arm("split_gmv30", split_arm("gmv30"))
arm("split_visit30", split_arm("visit30"))
arm("drop_dead", lambda seed: fit_predict(Xtr[tr_seg["visit180"]], lytr[tr_seg["visit180"]],
                                          Xev, seed))
q995 = np.quantile(lytr, 0.995)
arm("winsor995", lambda seed: fit_predict(Xtr, np.minimum(lytr, q995), Xev, seed))
arm("upweight_act", lambda seed: fit_predict(Xtr, lytr, Xev, seed,
                                             sw=np.where(tr_seg["gmv30"], 2.0, 1.0)))
