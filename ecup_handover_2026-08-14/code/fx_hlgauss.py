"""Binned-target classification probe (literature item #5): the last untested loss geometry.

log1p(y) -> 30 bins (bin 0 = exact zero, bins 1-29 = positive quantiles), LightGBM
multiclass, decode E[log1p y] = sum p_k * c_k with c_k = train mean of the bin. Hard
labels (true HL-Gauss soft targets need a custom objective — only worth it if this
probe shows life).

Kill criteria (from the digest): standalone >0.01 worse with corr >= 0.997 to the L2
baseline (nothing to blend), or half/half blend gain <= seed spread.

Usage:  python -u code/fx_hlgauss.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
NBINS = 30

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
Xev = X[SEL_EV]

pos = lytr > 0
qs = np.quantile(lytr[pos], np.linspace(0, 1, NBINS)[1:-1])
lab = np.where(pos, 1 + np.searchsorted(qs, lytr), 0).astype(np.int32)
centers = np.array([lytr[lab == k].mean() if (lab == k).any() else 0.0
                    for k in range(NBINS)])
print(f"bins: {NBINS}, zero-bin share {np.mean(lab==0):.3f}")

t0 = time.time()
m = lgb.LGBMClassifier(objective="multiclass", num_class=NBINS, n_estimators=200,
                       learning_rate=0.15, num_leaves=63, min_child_samples=100,
                       max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       reg_lambda=5.0, n_jobs=23, verbose=-1, random_state=42,
                       force_col_wise=True)
m.fit(Xtr, lab)
proba = m.predict_proba(Xev)
lp = proba @ centers
print(f"fitted ({time.time()-t0:.0f}s)")

s = rmsle(yev, np.expm1(np.clip(lp, 0, None)))
base = np.load(OUT / "fx_se_l2_s42.npy")
sb = rmsle(yev, np.expm1(np.clip(base, 0, None)))
corr = np.corrcoef(lp, base)[0, 1]
print(f"binned-decode RMSLE {s:.5f}   L2 seed42 {sb:.5f}   corr {corr:.5f}")

# half/half OLS blend with the L2 baseline
rng = np.random.default_rng(0)
half = rng.random(len(yev)) < 0.5
ly = np.log1p(yev)
A = np.stack([base, lp, np.ones_like(lp)], 1)
w = np.linalg.lstsq(A[half], ly[half], rcond=None)[0]
blend = A[~half] @ w
s_blend = rmsle(yev[~half], np.expm1(np.clip(blend, 0, None)))
s_base_half = rmsle(yev[~half], np.expm1(np.clip(base[~half], 0, None)))
# slope-only control on the same half split
A1 = np.stack([base, np.ones_like(base)], 1)
w1 = np.linalg.lstsq(A1[half], ly[half], rcond=None)[0]
s_slope = rmsle(yev[~half], np.expm1(np.clip(A1[~half] @ w1, 0, None)))
print(f"half/half: base {s_base_half:.5f}  base+slope {s_slope:.5f}  "
      f"blend {s_blend:.5f}  (blend weight on binned: {w[1]:+.3f})")
