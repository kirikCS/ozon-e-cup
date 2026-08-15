"""CatBoost SGLB virtual-ensemble probe (literature item #10).

Seed averaging is closed for CatBoost (t11: seeds corr 0.99766, averaging worse). SGLB
(langevin=True) injects gradient noise and lets one trained model emit a virtual ensemble
(staged predictions over tree-count suffixes). If the virtual members decorrelate below
the 0.997 seed floor, their mean could de-noise the CatBoost component the way seeds
de-noise LightGBM — the one stack member the seed trick cannot reach.

Instant kill: corr between virtual members >= 0.997, or virtual mean worse than plain.

Usage:  python -u code/fx_langevin.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
Xev = X[SEL_EV]

CB = dict(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0, border_count=64,
          loss_function="RMSE", verbose=0, thread_count=23, random_seed=42,
          bootstrap_type="Bernoulli", subsample=0.8)

t0 = time.time()
plain = CatBoostRegressor(**CB)
plain.fit(Xtr, lytr)
p_plain = plain.predict(Xev)
s_plain = rmsle(yev, np.expm1(np.clip(p_plain, 0, None)))
print(f"plain CatBoost: {s_plain:.5f}  ({time.time()-t0:.0f}s)")

t0 = time.time()
sg = CatBoostRegressor(**{**CB, "langevin": True, "diffusion_temperature": 250000})
sg.fit(Xtr, lytr)
vp = sg.virtual_ensembles_predict(Xev, prediction_type="VirtEnsembles",
                                  virtual_ensembles_count=5)
V = np.asarray(vp)[:, :, 0].T if np.asarray(vp).ndim == 3 else np.asarray(vp).T
print(f"SGLB fitted ({time.time()-t0:.0f}s), virtual members {V.shape}")

scs = [rmsle(yev, np.expm1(np.clip(v, 0, None))) for v in V]
mean_v = V.mean(0)
s_mean = rmsle(yev, np.expm1(np.clip(mean_v, 0, None)))
cc = np.corrcoef(V)
iu = np.triu_indices(len(V), 1)
print(f"virtual members RMSLE: {[round(x,5) for x in scs]}")
print(f"member corr: mean {cc[iu].mean():.5f}  min {cc[iu].min():.5f}   "
      f"(kill threshold 0.997)")
print(f"virtual mean {s_mean:.5f}  vs plain {s_plain:.5f}  delta {s_mean-s_plain:+.5f}")
print(f"corr(virtual mean, plain) {np.corrcoef(mean_v, p_plain)[0,1]:.5f}")
