"""Reproduce one reference model end to end, to prove the feature store works locally.

"LightGBM L2 on log1p" is the right probe: it is the plainest entry in the Tier-3 battery
(no BTYD features, no calibration, no two-stage), so a mismatch can only come from the
panels or the feature store. Published on the server: VAL 1.69101, HON 1.75150.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, build_features, ds)

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264, 236]
TR_HON = [258, 230, 202, 174, 146]
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))

t0 = time.time()
X, NAMES = build_features(ALL_A)
print(f"feature store built in {time.time()-t0:.0f}s")

Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)

import lightgbm as lgb
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")

EXPECT = {"VAL": 1.69101, "HON": 1.75150}
out = {}
for key, tr, ev, mask in [("VAL", TR_VAL, A, None), ("HON", TR_HON, AH, m_hon)]:
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(Y[a]) for a in tr])
    t0 = time.time()
    m = lgb.LGBMRegressor(**LGB)
    m.fit(Xtr, lytr)
    p = np.expm1(m.predict(X[ev]))
    yy, secs = Y[ev], time.time() - t0
    s = rmsle(yy, p) if mask is None else rmsle(yy[mask], p[mask])
    out[key] = s
    d = s - EXPECT[key]
    print(f"  {key}  anchor {ds(ev)}  RMSLE {s:.5f}   published {EXPECT[key]}   delta {d:+.5f}   ({secs:.0f}s)")
    np.save(Path("model_out") / f"verify_lgbm_{key.lower()}.npy", p.astype(np.float32))

ok = all(abs(out[k] - EXPECT[k]) < 3e-3 for k in EXPECT)
print("\nreproduced" if ok else "\nMISMATCH — panels or feature store differ from the server run")
sys.exit(0 if ok else 1)
