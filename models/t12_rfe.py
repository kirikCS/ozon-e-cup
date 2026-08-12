"""Recursive feature elimination, run separately per stack architecture.

296 features, of which the v2 block earns 11.4% of the gain and one feature alone earns 51.6%.
That is a strong hint that most of the matrix is noise the trees are spending splits on.

Method: repeatedly drop the weakest fraction of features by permutation-free gain importance,
refit, and score. Gain is measured on the training fit; the selection decision is made on the
VAL protocol, and the honest protocol is reported alongside as an independent check that the
shrinking set is not being tuned into the validation window.
"""
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from common import VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, record, build_features, OUT
from features_v2 import build_v2

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264, 236]
TR_HON = [258, 230, 202, 174, 160]
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))
KEEP_FRACS = [1.0, 0.75, 0.55, 0.40, 0.28, 0.20, 0.14, 0.10, 0.07]

X1, N1 = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
BT_NAMES = ["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"]
for a in ALL_A:
    X1[a] = np.concatenate([X1[a], BT[a]], 1)
X2, N2 = build_v2(ALL_A)
NAMES = np.array(N1 + BT_NAMES + N2)
X = {a: np.concatenate([X1[a], X2[a]], 1) for a in ALL_A}
del X1, X2
F = X[A].shape[1]
print(f"features: {F} ({len(N1)} v1 + {len(BT_NAMES)} btyd + {len(N2)} v2)")

Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)
PROTO = {"VAL": (TR_VAL, A, None), "HON": (TR_HON, AH, m_hon)}

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

def fit_two_stage(Xtr, ytr, lytr, Xev, **kw):
    pos = ytr > 0
    clf = lgb.LGBMClassifier(**dict(LGB)); clf.fit(Xtr, pos.astype(np.int8))
    reg = lgb.LGBMRegressor(**dict(LGB, objective="regression")); reg.fit(Xtr[pos], lytr[pos])
    p = np.expm1(clf.predict_proba(Xev)[:, 1] * np.clip(reg.predict(Xev), 0, None))
    # a feature matters to this model if it matters to either head
    gi = clf.booster_.feature_importance("gain"); gr = reg.booster_.feature_importance("gain")
    return p, gi / max(gi.sum(), 1) + gr / max(gr.sum(), 1)

def fit_cat(Xtr, ytr, lytr, Xev, **kw):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev)), np.asarray(m.get_feature_importance())

MODELS = [
    ("LightGBM DART", lambda *a, **k: fit_lgb(*a, extra={"boosting_type": "dart"}, n=250)),
    ("CatBoost RMSE", fit_cat),
    ("Two-stage LGBM", fit_two_stage),
    ("LightGBM L2", fit_lgb),
]

results = {}
for mname, fn in MODELS:
    print(f"\n=== RFE: {mname} ===")
    idx = np.arange(F)
    hist = []
    for frac in KEEP_FRACS:
        k = max(8, int(round(F * frac)))
        if k < len(idx):                                  # drop the weakest by gain from the last fit
            idx = idx[np.argsort(imp)[::-1][:k]]
        scores, imp_acc, secs = {}, None, 0.0
        for key, (tr, ev, mask) in PROTO.items():
            Xtr = np.concatenate([X[a][:, idx] for a in tr], 0)
            ytr = np.concatenate([Y[a] for a in tr]); lytr = np.log1p(ytr)
            t0 = time.time()
            p, gain = fn(Xtr, ytr, lytr, X[ev][:, idx])
            secs += time.time() - t0
            p = np.clip(p, 0, None)
            scores[key] = rmsle(Y[ev], p) if mask is None else rmsle(Y[ev][mask], p[mask])
            if key == "VAL":
                imp_acc = gain
            del Xtr
        imp = imp_acc
        hist.append((len(idx), scores["VAL"], scores["HON"]))
        print(f"  {len(idx):4d} признаков  VAL {scores['VAL']:.5f}  HON {scores['HON']:.5f}  ({secs:.0f}s)")
        results.setdefault(mname, []).append(
            {"n": int(len(idx)), "val": scores["VAL"], "hon": scores["HON"],
             "kept": NAMES[idx].tolist() if len(idx) <= 40 else None})
    best = min(hist, key=lambda r: r[1])
    full = hist[0]
    print(f"  лучший: {best[0]} признаков, VAL {best[1]:.5f} (полный набор {full[0]}: {full[1]:.5f}, "
          f"дельта {full[1]-best[1]:+.5f})")
    record(f"{mname} RFE-{best[0]}", "rfe", best[1], best[2], None,
           f"из {F} признаков; полный набор давал {full[1]:.5f}")

json.dump(results, open(OUT / "rfe_results.json", "w"), ensure_ascii=False, indent=1)
print("\ndone")
