"""Gate reliability diagnostic (literature item #6, first half — the free check).

The two-stage product P(y>0) * E[log1p y | y>0] is MSLE-optimal only if the gate is
calibrated. This bins the SELECT-eval gate scores and compares predicted vs realised
positive rates. If the reliability curve is already flat, the recalibration idea closes
without further work; if not, it reports where the distortion lives and the MSLE-optimal
power gamma as a first-order fix estimate.

Usage:  python -u code/fx_gate.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
pos = lytr > 0

t0 = time.time()
clf = lgb.LGBMClassifier(**LGB, random_state=42)
clf.fit(Xtr, pos.astype(np.int8))
pb = clf.predict_proba(X[SEL_EV])[:, 1]
reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=42)
reg.fit(Xtr[pos], lytr[pos])
mu = np.clip(reg.predict(X[SEL_EV]), 0, None)
print(f"fitted ({time.time()-t0:.0f}s)")

yb = (yev > 0).astype(np.float64)
print(f"\nreliability at anchor 348 (gate deciles): predicted vs realised P(y>0)")
qs = np.percentile(pb, np.arange(10, 100, 10))
bins = np.searchsorted(qs, pb)
for b in range(10):
    m = bins == b
    print(f"  bin {b}: pred {pb[m].mean():.4f}  real {yb[m].mean():.4f}  "
          f"diff {pb[m].mean()-yb[m].mean():+.4f}  n {m.sum()}")

base = rmsle(yev, np.expm1(pb * mu))
print(f"\ntwo-stage RMSLE as-is: {base:.5f}")
print("power-tempering sweep p -> p^g (applied to the gate only):")
for g in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
    s = rmsle(yev, np.expm1(np.clip(pb, 1e-9, 1) ** g * mu))
    print(f"  g={g:4.2f}: {s:.5f}  ({s-base:+.5f})")
