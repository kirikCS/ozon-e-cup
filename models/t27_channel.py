"""Boosting on the 296 existing features plus the catalog channel (v3).

Three runs, to separate whether the catalog columns carry signal or only decorrelation:
    ch_lgbm     LightGBM on [296 | v3]
    ch_catboost CatBoost on [296 | v3]
    ch_only     LightGBM on v3 alone — a deliberately weak model that shares almost no inputs
                with anything in the stack, which is the property the stack has been short of
"""
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from zoo_common import load, emit, TR_PROBE, TR_FINAL, FIT, PRED
from common import rmsle
import features_v3

X, Y, NAMES, TOP = load()
ANCH = sorted(set(TR_PROBE + TR_FINAL + [FIT, PRED]))
t0 = time.time()
V3, V3N = features_v3.build(ANCH)
print(f"v3 построен за {time.time()-t0:.0f}s")

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=700, learning_rate=0.07, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")

def feats(a, mode):
    return V3[a] if mode == "only" else np.concatenate([X[a], V3[a]], 1)

def run(label, fname, mode, fitter):
    t = time.time()
    Xtr = np.concatenate([feats(a, mode) for a in TR_PROBE], 0)
    ytr = np.log1p(np.concatenate([Y[a] for a in TR_PROBE]))
    p, imp = fitter(Xtr, ytr, feats(FIT, mode))
    del Xtr
    print(f"  {label}: probe {rmsle(Y[FIT], np.expm1(np.clip(p,0,13))):.5f} "
          f"({feats(FIT, mode).shape[1]} колонок, {time.time()-t:.0f}s)")
    if imp is not None:
        cn = (V3N if mode == "only" else NAMES + V3N)
        tot = imp.sum()
        v3share = imp[len(NAMES):].sum() / tot if mode != "only" else 1.0
        top = np.argsort(-imp)[:12]
        print(f"    доля важности у v3: {v3share*100:.1f}%")
        print("    топ: " + ", ".join(f"{cn[i]} {imp[i]/tot*100:.1f}%" for i in top))
    Xtr = np.concatenate([feats(a, mode) for a in TR_FINAL], 0)
    ytr = np.log1p(np.concatenate([Y[a] for a in TR_FINAL]))
    pf, _ = fitter(Xtr, ytr, feats(PRED, mode))
    del Xtr
    emit(np.clip(pf, 0, 13), fname)

def f_lgb(Xtr, lytr, Xev):
    m = lgb.LGBMRegressor(**LGB).fit(Xtr, lytr)
    return m.predict(Xev), m.booster_.feature_importance("gain")

def f_cat(Xtr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8).fit(Xtr, lytr)
    return m.predict(Xev), m.get_feature_importance()

print("\n=== каталожный канал ===")
run("LightGBM [296|v3]", "ch_lgbm", "both", f_lgb)
run("CatBoost [296|v3]", "ch_catboost", "both", f_cat)
run("LightGBM только v3", "ch_only", "only", f_lgb)
print("\ndone")
