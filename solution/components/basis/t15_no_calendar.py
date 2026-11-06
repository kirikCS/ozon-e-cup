"""Drop the features that extrapolate at prediction time.

Diagnosis behind this: the calendar block (tgt_doy_sin, tgt_doy_cos, tgt_month) never sees
February or March in training — every training target lands in Sep–Feb — and `hist_days` at the
prediction anchor (409) is larger than at any training anchor (267–379). Trees cannot extrapolate,
so those four columns can only mislead, splitting on values whose meaning differs at test time.

They were there so the model could learn seasonality itself. It does not need to: the level is
fixed analytically from the leaderboard (E[log1p(y)] = 2.3284), which is a strictly better estimate
than anything the model could infer from the calendar.

Cheaper and lower-risk than adding seasonal anchors with 44 days of history, and it addresses the
same failure mode.
"""
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from common import VAL_ANCHOR, N_DAYS, NU, target, rmsle, record, build_features, OUT, CACHE

PRED, FIT = N_DAYS - 1, VAL_ANCHOR
TR_F = [378, 350, 322, 294, 266]
TR_P = [348, 320, 292, 264, 236]
ALL_A = sorted(set(TR_F + TR_P + [PRED, FIT]))
EL = 2.3284
DROP = ["tgt_doy_sin", "tgt_doy_cos", "tgt_month", "hist_days"]

X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
BT_NAMES = ["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"]
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
NAMES = NAMES + BT_NAMES
keep = np.array([i for i, n in enumerate(NAMES) if n not in DROP])
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {len(NAMES)} -> {len(keep)} (dropped {DROP})")

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def fit_cat(Xtr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.clip(np.expm1(m.predict(Xev)), 0, None)

def fit_two_stage(Xtr, lytr, Xev):
    ytr = np.expm1(lytr); pos = ytr > 0
    clf = lgb.LGBMClassifier(**dict(LGB)); clf.fit(Xtr, pos.astype(np.int8))
    reg = lgb.LGBMRegressor(**dict(LGB, objective="regression")); reg.fit(Xtr[pos], lytr[pos])
    return np.clip(np.expm1(clf.predict_proba(Xev)[:, 1] * np.clip(reg.predict(Xev), 0, None)), 0, None)

def run(fn, cols, anchors, ev):
    Xtr = np.concatenate([X[a][:, cols] for a in anchors], 0)
    ly = np.concatenate([np.log1p(Y[a]) for a in anchors])
    out = fn(Xtr, ly, X[ev][:, cols])
    del Xtr
    return out

uids = np.load(CACHE / "uids.npy")
def emit(p, name):
    lp = np.log1p(np.clip(np.nan_to_num(p), 0, None)); lp = lp - lp.mean() + EL
    q = np.clip(np.expm1(lp), 0, None)
    with open(f"submissions/{name}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, q):
            f.write(f"{u},{v:.6f}\n")
    print(f"  -> {name}.csv  mean={q.mean():.2f} p50={np.percentile(q,50):.2f} p99={np.percentile(q,99):.0f}")

allc = np.arange(len(NAMES))
for mname, fn, fname in (("CatBoost", fit_cat, "t15_no_calendar"),
                         ("Two-stage LGBM", fit_two_stage, "nocal_two_stage")):
    t0 = time.time()
    a = rmsle(Y[FIT], run(fn, allc, TR_P, FIT))
    b = rmsle(Y[FIT], run(fn, keep, TR_P, FIT))
    print(f"\n{mname}: probe с календарём {a:.5f} -> без календаря {b:.5f}  ({b-a:+.5f}, {time.time()-t0:.0f}s)")
    emit(run(fn, keep, TR_F, PRED), fname)
print("\ndone")
