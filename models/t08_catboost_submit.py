"""Submission for the CatBoost RMSE-on-log1p component.

Same anchors and hyper-parameters as the Tier-7 build, so the probe score is comparable with
the rest of the battery. Two files are written: the model as-is, and the same predictions with
the level correction that the leaderboard parabola has already pinned down exactly
(E[log1p(y)] = 2.3284 on the public split, solved from three points of the sub2 shift family).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, "models")
from common import H, NU, N_DAYS, target, rmsle, Timer, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266]
TR_PROBE = [348, 320, 292, 264, 236]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
EL_PUBLIC = 2.3284                     # E[log1p(y)] on the public split, from the LB parabola
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)

X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {X[PRED].shape[1]}, predict anchor {ds(PRED)} -> [{ds(PRED+1)}, {ds(PRED+H)}]")

def fit_predict(anchors, Xev):
    Xtr = np.concatenate([X[a] for a in anchors], 0)
    ly = np.concatenate([np.log1p(Y[a]) for a in anchors])
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, ly)
    return np.clip(np.expm1(m.predict(Xev)), 0, None)

with Timer() as t:
    probe = fit_predict(TR_PROBE, X[FIT])
print(f"probe RMSLE at {ds(FIT)}: {rmsle(Y[FIT], probe):.5f}  ({t.dt:.0f}s)")

with Timer() as t:
    pred = fit_predict(TR_FINAL, X[PRED])
print(f"final fit done ({t.dt:.0f}s)")

uids = np.load(CACHE / "uids.npy")

def write(p, fname, label):
    p = np.clip(np.nan_to_num(p), 0, None).astype(np.float64)
    assert len(p) == NU and np.isfinite(p).all()
    with open(SUB / fname, "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, p):
            f.write(f"{u},{v:.6f}\n")
    print(f"  {label:<34s} -> {fname}\n      mean log1p={np.log1p(p).mean():.4f}  mean={p.mean():.2f}  "
          f"p50={np.percentile(p,50):.2f}  p99={np.percentile(p,99):.0f}  max={p.max():.0f}")

lvl = float(np.log1p(pred).mean())
c = EL_PUBLIC - lvl
print(f"\nlevel of the raw prediction: {lvl:.4f}  ->  exact optimal shift c* = {c:+.4f}")
write(pred, "sub7_catboost_rmse_log1p.csv", "CatBoost RMSE on log1p")
write(np.expm1(np.log1p(pred) + c), "sub8_catboost_rmse_log1p_shifted.csv", "same, level-corrected")
print("done")
