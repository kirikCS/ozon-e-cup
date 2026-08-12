"""Distributional boosting: models that fit the whole predictive distribution, not just its mean.

Caveat worth stating up front: under RMSLE the optimal point forecast is E[log1p(y)|x], so modelling
the spread cannot by itself beat a plain L2 fit in log space — the extra parameters do not move the
optimum. The reason to run these is decorrelation: natural-gradient and alternating-parameter fits
land on a different function than ordinary gradient boosting, and the stack pays for that.

Three implementations:
  NGBoost                  natural gradient over (mu, log sigma) of a Normal on log1p(y)
  CatBoost RMSEWithUncertainty   the same idea native to CatBoost, jointly fitting mean and variance
  LightGBM-LSS             alternating boosters for mu and log sigma against the Normal NLL,
                           which is what XGBoostLSS/LightGBMLSS do under the hood
All are CPU-only, so they run while the GPU is unavailable.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from zoo_common import load, run_pair, TR_PROBE, TR_FINAL, FIT, PRED

X, Y, NAMES, TOP = load()
SUBSET = np.sort(TOP[:150])          # NGBoost is O(n_features) per tree fit and very slow otherwise

# --------------------------------------------------------------------- NGBoost
def ngboost_fit(Xtr, ytr, Xev):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), min(250_000, len(Xtr)), replace=False)
    base = DecisionTreeRegressor(criterion="friedman_mse", max_depth=6,
                                 min_samples_leaf=200, random_state=0)
    m = NGBRegressor(Dist=Normal, Score=LogScore, Base=base, n_estimators=300,
                     learning_rate=0.05, minibatch_frac=0.5, col_sample=0.7,
                     natural_gradient=True, verbose=False, random_state=0)
    m.fit(Xtr[idx], np.log1p(ytr[idx]))
    return m.predict(Xev)                      # posterior mean in log1p space

# --------------------------------------------------- CatBoost RMSEWithUncertainty
def catboost_unc_fit(Xtr, ytr, Xev):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSEWithUncertainty",
                          verbose=0, thread_count=23, random_seed=42,
                          bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, np.log1p(ytr))
    return m.predict(Xev)[:, 0]                # column 0 = mean, column 1 = log variance

# ------------------------------------------------------------------ LightGBM-LSS
def lgbmlss_fit(Xtr, ytr, Xev, rounds=3, n_est=300):
    """alternating boosters for the two parameters of a Normal on log1p(y).

    NLL = 0.5*((y-mu)/s)^2 + log s
      d/dmu     = -(y-mu)/s^2          d2/dmu2     = 1/s^2
      d/dlog_s  = 1 - ((y-mu)/s)^2     d2/dlog_s2  = 2*((y-mu)/s)^2   (Fisher-style, kept positive)
    """
    import lightgbm as lgb
    ly = np.log1p(ytr)
    mu = np.full(len(ly), ly.mean())
    ls = np.full(len(ly), np.log(ly.std() + 1e-6))
    mu_ev = np.full(len(Xev), ly.mean())
    P = dict(n_estimators=n_est, learning_rate=0.06, num_leaves=63, min_child_samples=100,
             max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
             n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)
    for r in range(rounds):
        s2 = np.exp(2 * ls)
        def obj_mu(yt, pred):
            z = ly - (mu + pred)
            return (-z / s2).astype(np.float64), (1.0 / s2).astype(np.float64)
        m1 = lgb.LGBMRegressor(objective=obj_mu, **P)
        m1.fit(Xtr, np.zeros(len(ly)))
        mu = mu + m1.predict(Xtr)
        mu_ev = mu_ev + m1.predict(Xev)

        def obj_s(yt, pred):
            z2 = ((ly - mu) ** 2) / np.exp(2 * (ls + pred))
            return (1.0 - z2).astype(np.float64), np.maximum(2.0 * z2, 1e-3).astype(np.float64)
        m2 = lgb.LGBMRegressor(objective=obj_s, **P)
        m2.fit(Xtr, np.zeros(len(ly)))
        ls = np.clip(ls + m2.predict(Xtr), -6, 4)
    return mu_ev

JOBS = [
    ("CatBoost RMSEWithUncertainty", "zoo_catboost_unc", catboost_unc_fit, None),
    ("LightGBM-LSS (Normal, alternating)", "zoo_lgbmlss", lgbmlss_fit, None),
    ("NGBoost (Normal, natural gradient)", "zoo_ngboost", ngboost_fit, SUBSET),
]
print(f"=== distributional boosting: {len(JOBS)} моделей (CPU) ===")
for name, fname, fit, cols in JOBS:
    print(f"\n{name}")
    t0 = time.time()
    run_pair(name, fname, fit, cols)
    print(f"    всего {time.time()-t0:.0f}s")
print("\ndone")
