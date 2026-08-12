"""Does the v2 feature set move the ceiling?

Same models, same protocol, same anchors as Tier 3 — only the feature matrix changes. That makes
the probe comparison meaningful here, unlike cross-model probe comparisons, which the leaderboard
has already shown to be unreliable.
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from common import VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, record, build_features, OUT
from features_v2 import build_v2

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264, 236]
TR_HON = [258, 230, 202, 174, 160]      # all of these have BTYD features precomputed
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))

print("building v1 features ...")
X1, N1 = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
have_bt = all(a in BT for a in ALL_A)
print(f"  BTYD available for all anchors: {have_bt}"
      + ("" if have_bt else f" (missing {[a for a in ALL_A if a not in BT]}, skipping BTYD block)"))
if have_bt:
    for a in ALL_A:
        X1[a] = np.concatenate([X1[a], BT[a]], 1)
print("building v2 features ...")
X2, N2 = build_v2(ALL_A)
print(f"v1 = {X1[A].shape[1]}, v2 = {X2[A].shape[1]}, total = {X1[A].shape[1] + X2[A].shape[1]}")

Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)
PROTO = {"VAL": (TR_VAL, A, None), "HON": (TR_HON, AH, m_hon)}

def stack(anchors, which):
    if which == "v1":
        return np.concatenate([X1[a] for a in anchors], 0)
    if which == "v2":
        return np.concatenate([X2[a] for a in anchors], 0)
    return np.concatenate([np.concatenate([X1[a], X2[a]], 1) for a in anchors], 0)

def one(anchors, which):
    a = anchors[0]
    return np.concatenate([X1[a], X2[a]], 1) if which == "both" else (X1[a] if which == "v1" else X2[a])

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def f_lgb(extra=None, n=None):
    def f(Xtr, ytr, lytr, Xev):
        p = dict(LGB); p["objective"] = "regression"
        if extra: p.update(extra)
        if n: p["n_estimators"] = n
        m = lgb.LGBMRegressor(**p); m.fit(Xtr, lytr)
        return np.expm1(m.predict(Xev)), m
    return f

def f_two_stage(Xtr, ytr, lytr, Xev):
    pos = ytr > 0
    clf = lgb.LGBMClassifier(**dict(LGB)); clf.fit(Xtr, pos.astype(np.int8))
    reg = lgb.LGBMRegressor(**dict(LGB, objective="regression")); reg.fit(Xtr[pos], lytr[pos])
    return np.expm1(clf.predict_proba(Xev)[:, 1] * np.clip(reg.predict(Xev), 0, None)), reg

def f_cat(Xtr, ytr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev)), m

def run(name, fn, which):
    res, secs, model = {}, 0.0, None
    for key, (tr, ev, mask) in PROTO.items():
        Xtr = stack(tr, which)
        ytr = np.concatenate([Y[a] for a in tr]); lytr = np.log1p(ytr)
        t0 = time.time()
        p, model = fn(Xtr, ytr, lytr, one([ev], which))
        secs += time.time() - t0
        p = np.clip(p, 0, None)
        res[key] = rmsle(Y[ev], p) if mask is None else rmsle(Y[ev][mask], p[mask])
        del Xtr
    record(f"{name} [{which}]", f"featset-{which}", res["VAL"], res["HON"], secs, f"{one([A],which).shape[1]} features")
    return model

print("\n--- Tier 11: v1 vs v2 vs v1+v2 ---")
for which in ("v1", "both"):
    run("LightGBM L2 on log1p", f_lgb(), which)
    run("Two-stage LGBM", f_two_stage, which)
    run("CatBoost RMSE on log1p", f_cat, which)
    run("LightGBM DART", f_lgb({"boosting_type": "dart"}, n=250), which)
m = run("LightGBM L2 on log1p", f_lgb(), "v2")

# which of the new features actually earn their place
mdl = run("LightGBM L2 on log1p (importance run)", f_lgb(), "both")
imp = mdl.booster_.feature_importance("gain")
names = N1 + (["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"] if have_bt else []) + N2
order = np.argsort(imp)[::-1]
print("\nтоп-25 признаков по gain (v2 отмечены *):")
for i in order[:25]:
    star = "*" if names[i] in N2 else " "
    print(f"  {star} {names[i]:32s} {imp[i]/imp.sum()*100:6.2f}%")
share = sum(imp[i] for i in range(len(names)) if names[i] in N2) / imp.sum()
print(f"\nсуммарная доля gain у v2-признаков: {share*100:.1f}% (их {len(N2)} из {len(names)})")
print("done")
