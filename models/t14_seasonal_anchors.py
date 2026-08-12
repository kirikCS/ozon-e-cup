"""Train with anchors whose target windows fall in the season of the test window.

Every training anchor used so far has its target in Sep 2025 – Feb 2026. The competition window
is 14 Feb – 15 Mar, a season the models have never seen: the calendar features (tgt_month,
tgt_doy_sin/cos) are extrapolating, and the leaderboard has already shown that a set tuned on the
January trough transfers badly.

The only historical analogue sits at anchors 43/57/71 (targets 2025-02-14..03-15 and just after).
Those anchors carry 44–72 days of history, so their long windows are truncated — `hist_days` lets
the model discount them, and a weight variant is built to test how much emphasis they deserve.

This cannot be validated offline: any training set that would let us score anchor 43 honestly
would need targets ending before day 43, i.e. anchors with under two weeks of history. The probe
at 2026-01-14 is reported only to show what the augmentation costs on the trough window.
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from common import VAL_ANCHOR, N_DAYS, NU, target, rmsle, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, VAL_ANCHOR
TR_BASE_F = [378, 350, 322, 294, 266]
TR_BASE_P = [348, 320, 292, 264, 236]
SEASONAL = [43, 57, 71]                    # targets 2025-02-14..03-15, ..03-29, ..04-12
ALL_A = sorted(set(TR_BASE_F + TR_BASE_P + SEASONAL + [PRED, FIT]))
EL = 2.3284

X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
missing = [a for a in ALL_A if a not in BT]
print(f"BTYD missing for {missing}" if missing else "BTYD available for all anchors")
if not missing:
    for a in ALL_A:
        X[a] = np.concatenate([X[a], BT[a]], 1)
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {X[PRED].shape[1]}")
for a in SEASONAL:
    print(f"  anchor {ds(a)}: target {ds(a+1)}..{ds(a+30)}, history {a+1} days, "
          f"target mean log1p {np.log1p(Y[a]).mean():.4f}")
print(f"  anchor {ds(FIT)}: target mean log1p {np.log1p(Y[FIT]).mean():.4f}  (trough)")

from catboost import CatBoostRegressor

def fit(anchors, weights, Xev):
    Xtr = np.concatenate([X[a] for a in anchors], 0)
    ly = np.concatenate([np.log1p(Y[a]) for a in anchors])
    sw = np.concatenate([np.full(NU, weights.get(a, 1.0)) for a in anchors])
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, ly, sample_weight=sw)
    out = np.clip(np.expm1(m.predict(Xev)), 0, None)
    del Xtr
    return out

uids = np.load(CACHE / "uids.npy")
def emit(p, name):
    lp = np.log1p(np.clip(np.nan_to_num(p), 0, None))
    lp = lp - lp.mean() + EL
    q = np.clip(np.expm1(lp), 0, None)
    with open(f"submissions/{name}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, q):
            f.write(f"{u},{v:.6f}\n")
    print(f"  -> {name}.csv  mean={q.mean():.2f} p50={np.percentile(q,50):.2f} "
          f"p99={np.percentile(q,99):.0f} max={q.max():.0f}")

VARIANTS = [
    ("baseline (no seasonal anchors)", [], {}, None),
    ("seasonal, weight 1.0", SEASONAL, {}, "seasonal_cat_w1"),
    ("seasonal, weight 2.0", SEASONAL, {a: 2.0 for a in SEASONAL}, "seasonal_cat_w2"),
    ("seasonal, weight 0.5", SEASONAL, {a: 0.5 for a in SEASONAL}, "seasonal_cat_w05"),
]
for label, extra, wts, fname in VARIANTS:
    t0 = time.time()
    pr = fit(TR_BASE_P + extra, wts, X[FIT])
    print(f"\n{label}: probe RMSLE {rmsle(Y[FIT], pr):.5f}  ({time.time()-t0:.0f}s)")
    if fname:
        emit(fit(TR_BASE_F + extra, wts, X[PRED]), fname)
print("\ndone")
