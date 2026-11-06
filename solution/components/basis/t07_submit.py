"""Build the three submissions for 2026-02-14 .. 2026-03-15.

Anchors
  PRED = 408 (2026-02-13)  -> target 2026-02-14 .. 2026-03-15, the competition window
  FIT  = 378 (2026-01-14)  -> target 2026-01-15 .. 2026-02-13, fully observed

Two model fits per component:
  * FINAL  trained on anchors ending at 378, used to predict at 408. Anchor 378 is usable now:
    its target window is entirely inside the training data.
  * PROBE  trained on anchors ending at 348 (exactly the validated Tier-3 setup), used to predict
    at 378. The blend weights and the isotonic curve are learned from these, so they never see
    the window the FINAL models were trained on.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "models")
import btyd_lib as BL
from common import H, NU, N_DAYS, panel, target, rmsle, Timer, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266]
TR_PROBE = [348, 320, 292, 264, 236]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
SUB_DIR = Path("submissions"); SUB_DIR.mkdir(exist_ok=True)

print(f"predict anchor {ds(PRED)} -> [{ds(PRED+1)}, {ds(PRED+H)}]")
print(f"blend/isotonic fitted at anchor {ds(FIT)} -> [{ds(FIT+1)}, {ds(FIT+H)}]")

# ------------------------------------------------------------------ BTYD features
G = panel("gmv"); V = panel("visit")
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
BUY = G > 0
csB = np.concatenate([np.zeros((NU, 1), np.int64), BUY.astype(np.int64)], 1).cumsum(1)
wG = lambda a, b: csG[:, max(b, 0) + 1] - csG[:, max(a, 0)]
SUBS = np.random.default_rng(0).choice(NU, 25000, replace=False)
K, KAP, DEC = 10, 0.5, 0.7          # EB log-block parameters tuned in Tier 2b

def rfm(a):
    sub = V[:, :a + 1] > 0
    first = np.argmax(sub, axis=1)
    bsub = BUY[:, :a + 1]
    has = bsub.any(axis=1)
    last_buy = a - np.argmax(bsub[:, ::-1], axis=1)
    n_buy = csB[:, a + 1]
    x = np.maximum(n_buy - 1, 0).astype(np.float64)
    T = (a - first).astype(np.float64)
    t_x = np.minimum(np.where(has, last_buy - first, 0).astype(np.float64), T)
    mval = np.where(n_buy > 0, csG[:, a + 1] / np.maximum(n_buy, 1), 0.0)
    return x, t_x, T, n_buy, mval

def eb_logblocks(a):
    L = np.stack([np.log1p(wG(a - 30 * i - 29, a - 30 * i)) for i in range(K)], 1)
    w = DEC ** np.arange(K)
    pop = (L * w).sum() / (w.sum() * NU)
    return np.expm1(((L * w).sum(1) + KAP * pop) / (w.sum() + KAP))

BT = {}
try:
    BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
except Exception:
    pass
missing = [a for a in ALL_A if a not in BT]
if missing:
    print(f"computing BTYD features for {missing}")
    for a in sorted(missing):
        x, t_x, T, nb, mv = rfm(a)
        par = BL.bgnbd_fit(x[SUBS], t_x[SUBS], T[SUBS])
        gpar = BL.gg_fit(nb, mv)
        BT[a] = np.stack([BL.bgnbd_expected(par, x, t_x, T, H), BL.bgnbd_palive(par, x, t_x, T),
                          BL.gg_expected(gpar, nb, mv), (csB[:, a + 1] + 1.0) / (a + 1 + 20.0),
                          np.log1p(eb_logblocks(a))], 1).astype(np.float32)
        print(f"  {ds(a)} done")
    pickle.dump(BT, open(OUT / "btyd_feats.pkl", "wb"))

# ------------------------------------------------------------------ features
print("building features ...")
with Timer() as t:
    X, NAMES = build_features(ALL_A)
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
print(f"  {X[PRED].shape[1]} features, {t.dt:.0f}s")
Y = {a: target(a) for a in ALL_A if a != PRED}

def stack(anchors):
    return np.concatenate([X[a] for a in anchors], 0)

# ------------------------------------------------------------------ components
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)
XGBP = dict(n_estimators=600, learning_rate=0.08, max_depth=7, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=23, max_bin=64,
            tree_method="hist", random_state=42)

def f_lgb(extra=None, n=None):
    def f(Xtr, ytr, lytr, Xev):
        p = dict(LGB); p["objective"] = "regression"
        if extra: p.update(extra)
        if n: p["n_estimators"] = n
        m = lgb.LGBMRegressor(**p); m.fit(Xtr, lytr)
        return np.expm1(m.predict(Xev))
    return f

def f_two_stage(Xtr, ytr, lytr, Xev):
    pos = ytr > 0
    cp = dict(LGB)
    clf = lgb.LGBMClassifier(**cp); clf.fit(Xtr, pos.astype(np.int8))
    p_buy = clf.predict_proba(Xev)[:, 1]
    rp = dict(LGB); rp["objective"] = "regression"
    reg = lgb.LGBMRegressor(**rp); reg.fit(Xtr[pos], lytr[pos])
    return np.expm1(p_buy * np.clip(reg.predict(Xev), 0, None))

def f_xgb(Xtr, ytr, lytr, Xev):
    m = xgb.XGBRegressor(objective="reg:squarederror", **XGBP); m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev))

def f_cat(Xtr, ytr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev))

# the eight components that made the Tier-6 blend, in validation-score order
COMPONENTS = [
    ("LightGBM DART on log1p", f_lgb({"boosting_type": "dart"}, n=250)),
    ("Two-stage LGBM + BTYD feats", f_two_stage),
    ("LightGBM L2 on log1p + BTYD feats", f_lgb()),
    ("LightGBM L2 log1p deep", f_lgb({"num_leaves": 255, "learning_rate": 0.05, "max_bin": 127}, n=1200)),
    ("CatBoost RMSE on log1p", f_cat),
    ("XGBoost L2 on log1p", f_xgb),
]

Xf, yf, lyf = stack(TR_FINAL), np.concatenate([Y[a] for a in TR_FINAL]), np.concatenate([np.log1p(Y[a]) for a in TR_FINAL])
Xp, yp, lyp = stack(TR_PROBE), np.concatenate([Y[a] for a in TR_PROBE]), np.concatenate([np.log1p(Y[a]) for a in TR_PROBE])
print(f"final train {Xf.shape}, probe train {Xp.shape}")

FINAL, PROBE = {}, {}
for name, fn in COMPONENTS:
    with Timer() as t:
        FINAL[name] = np.clip(fn(Xf, yf, lyf, X[PRED]), 0, None)
        PROBE[name] = np.clip(fn(Xp, yp, lyp, X[FIT]), 0, None)
    print(f"  {name:<44s} probe RMSLE {rmsle(Y[FIT], PROBE[name]):.5f}  ({t.dt:.0f}s)")

# ------------------------------------------------------------------ submissions
uids = np.load(CACHE / "uids.npy")

def write(pred, fname, label):
    pred = np.clip(np.nan_to_num(pred), 0, None).astype(np.float64)
    assert len(pred) == NU and np.isfinite(pred).all()
    path = SUB_DIR / fname
    with open(path, "w") as f:
        f.write("user_id,predict\n")
        for u, p in zip(uids, pred):
            f.write(f"{u},{p:.6f}\n")
    print(f"  {label:<46s} -> {fname}  mean={pred.mean():.2f} zeros={(pred==0).mean():.1%} "
          f"p50={np.percentile(pred,50):.1f} p99={np.percentile(pred,99):.0f} max={pred.max():.0f}")
    return path

print("\nwriting submissions:")
write(FINAL["LightGBM DART on log1p"], "t07_submit_lgbm_dart.csv", "1. LightGBM DART on log1p")
write(FINAL["Two-stage LGBM + BTYD feats"], "t07_submit_btyd.csv", "2. Two-stage LGBM + BTYD")

# blend: NNLS weights + isotonic curve learned at the FIT anchor, applied to the FINAL predictions
names = [n for n, _ in COMPONENTS]
Mp = np.stack([np.log1p(PROBE[n]) for n in names], 1)
Mf = np.stack([np.log1p(FINAL[n]) for n in names], 1)
w, _ = nnls(Mp, np.log1p(Y[FIT]))
print("\nblend weights (fitted at anchor {}):".format(ds(FIT)))
for n, wi in zip(names, w):
    print(f"    {wi:6.3f}  {n}")
blend_probe = Mp @ w
iso = IsotonicRegression(out_of_bounds="clip").fit(blend_probe, np.log1p(Y[FIT]))
print(f"  blend probe RMSLE          {rmsle(Y[FIT], np.expm1(blend_probe)):.5f}")
print(f"  blend + isotonic probe     {rmsle(Y[FIT], np.expm1(iso.predict(blend_probe))):.5f}")
write(np.expm1(iso.predict(Mf @ w)), "sub3_nnls_isotonic_blend.csv", "3. Isotonic calibration of NNLS blend")
print("\ndone")
