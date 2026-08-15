"""Two untested model-internal levers, screened on SELECT (2026-08-12 night).

1. Training-anchor count. Components train on 5 anchors spaced 28d; history allows more.
   Variants: 5 (base), 8, 10 anchors — all target windows end <= eval anchor 348 (the
   oldest anchor 66 has 66 days of history; drift vs data-volume tradeoff is the question).
2. More trees at lower LR: 1200@0.04, 2400@0.02 — boosting-noise reduction partially
   orthogonal to seed averaging.

Both are model-internal (§4): if either survives here + VAL/HON, it chains onto the
freshly measured LB anchors of the local-pipeline vectors without any pipeline-drift
assumption. 3 seeds per variant; base seed sd on this screen is ~0.0004, so only effects
clearing ~0.001 are considered real.

Usage:  python -u code/fx_anchors.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, OUT

SEL_EV = 348
ANCH5 = [318, 290, 262, 234, 206]
ANCH8 = ANCH5 + [178, 150, 122]
ANCH10 = ANCH8 + [94, 66]
SEEDS = [42, 101, 777]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")

anchors = sorted(set(ANCH10 + [SEL_EV]))
X, names = build_features(anchors)
yev = target(SEL_EV)
Xev = X[SEL_EV]

VARIANTS = [
    ("base 5a", ANCH5, {}),
    ("8 anchors", ANCH8, {}),
    ("10 anchors", ANCH10, {}),
    ("n1200 lr.04", ANCH5, {"n_estimators": 1200, "learning_rate": 0.04}),
    ("n2400 lr.02", ANCH5, {"n_estimators": 2400, "learning_rate": 0.02}),
    ("8a n1200 lr.04", ANCH8, {"n_estimators": 1200, "learning_rate": 0.04}),
]

base_scores = {}
for label, tr, extra in VARIANTS:
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(target(a)) for a in tr])
    scores = {}
    for seed in SEEDS:
        t0 = time.time()
        params = dict(LGB); params.update(extra)
        m = lgb.LGBMRegressor(**params, random_state=seed)
        m.fit(Xtr, lytr)
        lp = m.predict(Xev)
        scores[seed] = rmsle(yev, np.expm1(np.clip(lp, 0, None)))
        np.save(OUT / f"fx_anch_{label.replace(' ', '_')}_s{seed}.npy", lp.astype(np.float32))
        d = f"  d {scores[seed]-base_scores[seed]:+.5f}" if base_scores else ""
        print(f"  {label:<16} seed {seed:>4}: {scores[seed]:.5f}{d}  ({time.time()-t0:.0f}s)",
              flush=True)
    if not base_scores:
        base_scores = scores
    ens = np.mean([np.load(OUT / f"fx_anch_{label.replace(' ', '_')}_s{s}.npy")
                   for s in SEEDS], 0)
    e = rmsle(yev, np.expm1(np.clip(ens, 0, None)))
    d = np.mean([scores[s] - base_scores[s] for s in SEEDS])
    print(f"  {label}: mean delta {d:+.5f}   3-seed ens {e:.5f}", flush=True)
