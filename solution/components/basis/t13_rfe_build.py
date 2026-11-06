"""Replay the RFE path to each model's optimum, dump the surviving feature set, then train that
model on the final anchors and emit a submission.

The elimination path is deterministic (fixed seeds, importances taken from the VAL fit only), so
replaying the VAL protocol alone reproduces exactly the sets that Tier 12 selected.
"""
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from common import VAL_ANCHOR, N_DAYS, NU, target, rmsle, build_features, OUT, CACHE
from features_v2 import build_v2

PRED, FIT = N_DAYS - 1, VAL_ANCHOR
TR_PROBE = [348, 320, 292, 264, 236]
TR_FINAL = [378, 350, 322, 294, 266]
ALL_A = sorted(set(TR_PROBE + TR_FINAL + [PRED, FIT]))
KEEP_FRACS = [1.0, 0.75, 0.55, 0.40, 0.28, 0.20, 0.14, 0.10, 0.07]
EL = 2.3284
BEST = {"LightGBM DART": 83, "CatBoost RMSE": 41, "Two-stage LGBM": 41, "LightGBM L2": 83}

X1, N1 = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
BT_NAMES = ["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"]
for a in ALL_A:
    X1[a] = np.concatenate([X1[a], BT[a]], 1)
X2, N2 = build_v2(ALL_A)
NAMES = np.array(N1 + BT_NAMES + N2)
X = {a: np.concatenate([X1[a], X2[a]], 1) for a in ALL_A}
del X1, X2
F = X[PRED].shape[1]
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {F}")

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def fit_lgb(Xtr, ytr, lytr, Xev, extra=None, n=None):
    p = dict(LGB, objective="regression")
    if extra: p.update(extra)
    if n: p["n_estimators"] = n
    m = lgb.LGBMRegressor(**p); m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev)), m.booster_.feature_importance("gain")

def fit_two_stage(Xtr, ytr, lytr, Xev):
    pos = ytr > 0
    clf = lgb.LGBMClassifier(**dict(LGB)); clf.fit(Xtr, pos.astype(np.int8))
    reg = lgb.LGBMRegressor(**dict(LGB, objective="regression")); reg.fit(Xtr[pos], lytr[pos])
    p = np.expm1(clf.predict_proba(Xev)[:, 1] * np.clip(reg.predict(Xev), 0, None))
    gi = clf.booster_.feature_importance("gain"); gr = reg.booster_.feature_importance("gain")
    return p, gi / max(gi.sum(), 1) + gr / max(gr.sum(), 1)

def fit_cat(Xtr, ytr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev)), np.asarray(m.get_feature_importance())

MODELS = {
    "LightGBM DART": (lambda *a: fit_lgb(*a, extra={"boosting_type": "dart"}, n=250), "rfe_dart"),
    "CatBoost RMSE": (fit_cat, "t13_rfe_catboost"),
    "Two-stage LGBM": (fit_two_stage, "rfe_two_stage"),
    "LightGBM L2": (fit_lgb, "t13_rfe_lgbm"),
}

def train(anchors, idx, fn, Xev):
    Xtr = np.concatenate([X[a][:, idx] for a in anchors], 0)
    ytr = np.concatenate([Y[a] for a in anchors])
    out = fn(Xtr, ytr, np.log1p(ytr), Xev)
    del Xtr
    return out

uids = np.load(CACHE / "uids.npy")
selected = {}
for mname, (fn, fname) in MODELS.items():
    target_n = BEST[mname]
    print(f"\n=== {mname} -> {target_n} признаков ===")
    idx = np.arange(F); imp = None
    t0 = time.time()
    for frac in KEEP_FRACS:
        k = max(8, int(round(F * frac)))
        if k < len(idx):
            idx = idx[np.argsort(imp)[::-1][:k]]
        if len(idx) == target_n:
            break
        _, imp = train(TR_PROBE, idx, fn, X[FIT][:, idx])
    p_probe, _ = train(TR_PROBE, idx, fn, X[FIT][:, idx])
    print(f"  probe RMSLE {rmsle(Y[FIT], np.clip(p_probe, 0, None)):.5f}  ({time.time()-t0:.0f}s)")
    selected[mname] = NAMES[idx].tolist()

    p, _ = train(TR_FINAL, idx, fn, X[PRED][:, idx])
    lp = np.log1p(np.clip(np.nan_to_num(p), 0, None))
    lp = lp - lp.mean() + EL
    pred = np.clip(np.expm1(lp), 0, None)
    with open(f"submissions/{fname}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, pred):
            f.write(f"{u},{v:.6f}\n")
    print(f"  -> {fname}.csv  mean={pred.mean():.2f} p50={np.percentile(pred,50):.2f} "
          f"p99={np.percentile(pred,99):.0f}")

json.dump(selected, open(OUT / "rfe_selected.json", "w"), ensure_ascii=False, indent=1)

sets = {k: set(v) for k, v in selected.items()}
core = set.intersection(*sets.values())
union = set.union(*sets.values())
print(f"\nобщее ядро всех четырёх моделей: {len(core)} признаков из {len(union)} в объединении")
for n in sorted(core):
    tag = "v2" if n in N2 else ("btyd" if n in BT_NAMES else "v1")
    print(f"  [{tag:4s}] {n}")
print("\nразмеры наборов:", {k: len(v) for k, v in selected.items()})
print("done")
