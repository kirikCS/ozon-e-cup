"""catboost v3: extended anchors + slower boosting + no calendar columns.

Lever confirmed for CatBoost on three protocols (2026-08-13): SELECT -0.0003,
VAL -0.0102 (!), HON -0.0010. Negative control reproduced: extension WITHOUT droptgt
explodes VAL to 2.005 — CatBoost's out-of-range month extrapolation is even worse than
LightGBM's. Single seed (t11: CatBoost seeds correlate 0.9977, averaging useless).

PROBE pass runs BOTH configs at 378 for the chain estimate; FINAL predicts at 408.

Outputs: submissions/catboost_v3.csv, model_out/t19_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import N_DAYS, target, rmsle, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266, 238, 210, 182, 154, 126]
TR_PROBE = [348, 320, 292, 264, 236, 208, 180, 152, 124, 96]
TR_FINAL5 = TR_FINAL[:5]
TR_PROBE5 = TR_PROBE[:5]
DROP = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}
EL = 2.3284

CB = dict(depth=7, l2_leaf_reg=5.0, border_count=64, loss_function="RMSE", verbose=0,
          thread_count=23, random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)

ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
X, NAMES = build_features(ALL_A)
keep = np.array([i for i, n in enumerate(NAMES) if n not in DROP])
Y = {a: target(a) for a in ALL_A if a != PRED}


def fit_predict(tr, ev, iters, lr, cols):
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(Y[a]) for a in tr])
    Xe = X[ev]
    if cols is not None:
        Xtr = Xtr[:, cols]; Xe = Xe[:, cols]
    m = CatBoostRegressor(iterations=iters, learning_rate=lr, **CB)
    m.fit(Xtr, lytr)
    return m.predict(Xe)              # log1p scale


t0 = time.time()
p_base = fit_predict(TR_PROBE5, FIT, 800, 0.09, None)
s_base = rmsle(Y[FIT], np.expm1(np.clip(p_base, 0, None)))
print(f"probe base (5a, 800it): {s_base:.5f} ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time()
p_v3 = fit_predict(TR_PROBE, FIT, 1600, 0.045, keep)
s_v3 = rmsle(Y[FIT], np.expm1(np.clip(p_v3, 0, None)))
print(f"probe v3 (10a, 1600it, droptgt): {s_v3:.5f} ({time.time()-t0:.0f}s)", flush=True)
json.dump({"probe_base": s_base, "probe_v3": s_v3}, open(OUT / "t19_probe.json", "w"), indent=1)

t0 = time.time()
lf = fit_predict(TR_FINAL, PRED, 1600, 0.045, keep)
print(f"final done ({time.time()-t0:.0f}s)", flush=True)
lf = np.asarray(lf)
lf = lf - lf.mean() + EL
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open(Path("submissions") / "catboost_v3.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"catboost_v3.csv: mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f} "
      f"p50={np.percentile(pred,50):.2f} max={pred.max():.0f}")
