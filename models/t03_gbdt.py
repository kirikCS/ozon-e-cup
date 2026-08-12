"""Tier 3: gradient boosting battery.
Two independent protocols so that neither evaluation can leak:
  VAL  — train on anchors whose target windows end <= 378, evaluate at anchor 378 (all 250k users, LB-like)
  HON  — train on anchors whose target windows end <= 288, evaluate at anchor 288 (selection rule replicated)
"""
import pickle
import sys
import numpy as np

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, H, NU, panel, target, cohort_rule,
                    rmsle, record, Timer, ds, build_features, OUT)

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264, 236]
TR_HON = [258, 230, 202, 174, 146]
CAL_VAL = 348          # held-out anchor used only for post-hoc calibration of VAL models
CAL_HON = 258
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))

print("building features ...")
with Timer() as t:
    X, NAMES = build_features(ALL_A)
print(f"  {len(NAMES)} features, {t.dt:.0f}s")

Y = {a: target(a) for a in ALL_A}
LY = {a: np.log1p(Y[a]) for a in ALL_A}
m_hon = cohort_rule(AH)

BT = None
try:
    with open(OUT / "btyd_feats.pkl", "rb") as f:
        BT = pickle.load(f)
    print(f"  BTYD features available for {len(BT)} anchors")
except Exception as e:
    print("  BTYD features not available:", e)

def stack(anchors, with_btyd=False):
    xs = []
    for a in anchors:
        blk = X[a]
        if with_btyd and BT is not None and a in BT:
            blk = np.concatenate([blk, BT[a]], axis=1)
        xs.append(blk)
    return np.concatenate(xs, 0)

def names_of(with_btyd):
    return NAMES + (["btyd_ex", "btyd_palive", "eb_p", "eb_logspend"] if with_btyd and BT else [])

PROTO = {
    "VAL": dict(tr=TR_VAL, ev=A, mask=None, cal=CAL_VAL),
    "HON": dict(tr=TR_HON, ev=AH, mask=m_hon, cal=CAL_HON),
}

def run(name, family, fit_predict, with_btyd=False, calibrate=False, notes=""):
    """fit_predict(Xtr, ytr_raw, ytr_log, Xev) -> raw-scale prediction.
    When calibrate=True the model is fitted once and asked for predictions on both the evaluation
    anchor and the held-out calibration anchor in a single call."""
    res = {}
    secs = 0
    for key, cfg in PROTO.items():
        tr = [a for a in cfg["tr"] if not calibrate or a != cfg["cal"]]
        if with_btyd and BT is not None:
            tr = [a for a in tr if a in BT]
            if cfg["ev"] not in BT:
                return None
        Xtr = stack(tr, with_btyd)
        ytr = np.concatenate([Y[a] for a in tr])
        lytr = np.concatenate([LY[a] for a in tr])
        Xev = stack([cfg["ev"]], with_btyd)
        if calibrate:
            Xev = np.concatenate([Xev, stack([cfg["cal"]], with_btyd)], 0)
        with Timer() as t:
            p = fit_predict(Xtr, ytr, lytr, Xev)
        secs += t.dt
        if calibrate:
            p, pc = p[:NU], p[NU:]
            best = (1e9, (1.0, 0.0))
            for b in np.linspace(0.5, 1.4, 37):
                for c in np.linspace(-0.4, 0.8, 25):
                    s = rmsle(Y[cfg["cal"]], np.expm1(b * np.log1p(np.clip(pc, 0, None)) + c))
                    if s < best[0]: best = (s, (b, c))
            b, c = best[1]
            p = np.expm1(b * np.log1p(np.clip(p, 0, None)) + c)
            notes_ = notes + f" | calib b={b:.2f} c={c:.2f}"
        else:
            notes_ = notes
        yy = Y[cfg["ev"]]
        mm = cfg["mask"]
        res[key] = rmsle(yy, p) if mm is None else rmsle(yy[mm], p[mm])
        if key == "VAL":
            np.save(OUT / f"pred_{name.replace('/', '_').replace(' ', '_')[:60]}.npy", p.astype(np.float32))
    return record(name, family, res["VAL"], res["HON"], secs, notes_ if calibrate else notes)

# ==================================================================== LightGBM
import lightgbm as lgb
# max_bin=63 instead of the 255 default is the single biggest lever here: histogram building
# dominates on 1.25M x 165, and the accuracy cost on features this smooth is negligible.
LGB_BASE = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
                max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def lgbm(objective, on_log, extra=None, n=None):
    def f(Xtr, ytr, lytr, Xev):
        p = dict(LGB_BASE); p["objective"] = objective
        if extra: p.update(extra)
        if n: p["n_estimators"] = n
        m = lgb.LGBMRegressor(**p)
        m.fit(Xtr, lytr if on_log else ytr)
        pr = m.predict(Xev)
        return np.expm1(pr) if on_log else pr
    return f

print("--- Tier 3: gradient boosting ---")
run("LightGBM L2 on log1p", "gbdt", lgbm("regression", True), notes="direct RMSLE optimisation")
run("LightGBM L2 on log1p + BTYD feats", "gbdt", lgbm("regression", True), with_btyd=True)
run("LightGBM Tweedie (raw)", "gbdt", lgbm("tweedie", False, {"tweedie_variance_power": 1.2}))
run("LightGBM Tweedie + log-calibration", "gbdt", lgbm("tweedie", False, {"tweedie_variance_power": 1.2}), calibrate=True)
run("LightGBM Poisson (raw)", "gbdt", lgbm("poisson", False))
run("LightGBM Poisson + log-calibration", "gbdt", lgbm("poisson", False), calibrate=True)
run("LightGBM Huber on log1p", "gbdt", lgbm("huber", True, {"alpha": 1.0}))
run("LightGBM quantile 0.5 on log1p", "gbdt", lgbm("quantile", True, {"alpha": 0.5}))
run("LightGBM MAE on log1p", "gbdt", lgbm("regression_l1", True))
run("LightGBM L2 log1p, deep (255 leaves, 1200 trees)", "gbdt",
    lgbm("regression", True, {"num_leaves": 255, "learning_rate": 0.05, "max_bin": 127}, n=1200))
run("LightGBM DART on log1p", "gbdt", lgbm("regression", True, {"boosting_type": "dart"}, n=250))

# ----------------------------------------------------- two-stage (exact for RMSLE)
def two_stage(clf_params=None, reg_params=None, lib="lgb"):
    def f(Xtr, ytr, lytr, Xev):
        pos = ytr > 0
        cp = dict(LGB_BASE); cp.pop("objective", None); cp.update(clf_params or {})
        clf = lgb.LGBMClassifier(**cp)
        clf.fit(Xtr, pos.astype(np.int8))
        p_buy = clf.predict_proba(Xev)[:, 1]
        rp = dict(LGB_BASE); rp["objective"] = "regression"; rp.update(reg_params or {})
        reg = lgb.LGBMRegressor(**rp)
        reg.fit(Xtr[pos], lytr[pos])
        mu = reg.predict(Xev)
        return np.expm1(p_buy * np.clip(mu, 0, None))
    return f

run("Two-stage LGBM: P(buy) x E[log1p | buy]", "two-stage", two_stage(),
    notes="exact identity E[log1p y] = P(y>0)*E[log1p y | y>0]")
run("Two-stage LGBM + BTYD feats", "two-stage", two_stage(), with_btyd=True)
run("Two-stage LGBM + post calibration", "two-stage", two_stage(), calibrate=True)

# ==================================================================== XGBoost
import xgboost as xgb
XGB_BASE = dict(n_estimators=600, learning_rate=0.08, max_depth=7, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=23, max_bin=64,
                tree_method="hist", random_state=42)

def xgbm(objective, on_log, extra=None):
    def f(Xtr, ytr, lytr, Xev):
        p = dict(XGB_BASE); p["objective"] = objective
        if extra: p.update(extra)
        m = xgb.XGBRegressor(**p)
        m.fit(Xtr, lytr if on_log else ytr)
        pr = m.predict(Xev)
        return np.expm1(pr) if on_log else pr
    return f

run("XGBoost L2 on log1p", "gbdt", xgbm("reg:squarederror", True))
run("XGBoost reg:squaredlogerror (raw)", "gbdt", xgbm("reg:squaredlogerror", False))
run("XGBoost Tweedie + log-calibration", "gbdt", xgbm("reg:tweedie", False, {"tweedie_variance_power": 1.2}), calibrate=True)

# ==================================================================== CatBoost
from catboost import CatBoostRegressor
def cbm(loss, on_log, extra=None):
    def f(Xtr, ytr, lytr, Xev):
        p = dict(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0, border_count=64,
                 loss_function=loss, verbose=0, thread_count=23, random_seed=42,
                 bootstrap_type="Bernoulli", subsample=0.8)
        if extra: p.update(extra)
        m = CatBoostRegressor(**p)
        m.fit(Xtr, lytr if on_log else ytr)
        pr = m.predict(Xev)
        return np.expm1(pr) if on_log else pr
    return f

run("CatBoost RMSE on log1p", "gbdt", cbm("RMSE", True))
run("CatBoost Tweedie + log-calibration", "gbdt", cbm("Tweedie:variance_power=1.2", False), calibrate=True)
run("CatBoost Quantile 0.5 on log1p", "gbdt", cbm("Quantile:alpha=0.5", True))

# ==================================================================== linear
from sklearn.linear_model import Ridge
def ridge_log():
    def f(Xtr, ytr, lytr, Xev):
        Xl = np.log1p(np.clip(Xtr, 0, None)); Xe = np.log1p(np.clip(Xev, 0, None))
        mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
        m = Ridge(alpha=10.0)
        m.fit((Xl - mu) / sd, lytr)
        return np.expm1(m.predict((Xe - mu) / sd))
    return f
run("Ridge on log-features -> log1p target", "linear", ridge_log())

print("Tier 3 done")
