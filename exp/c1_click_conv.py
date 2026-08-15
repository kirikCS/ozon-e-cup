"""Decompose the target into clicks x conversion x value — and stop hard-coding the product.

r5 established the asymmetry that makes this worth doing: log-clicks are predicted at RMSLE
0.47 (visits) against 1.74 for GMV itself. There is a lot of accuracy in the click factor. The
naive product threw all of it away and scored 2.38-2.63, for two reasons that are separable:

  ZEROS. Clicks do not vanish when purchases do. Visits are near-universal, so the product
  predicted positive GMV for the 46% who buy nothing. Fixed by putting the decomposition on the
  CONDITIONAL part only, where the exact identity applies:
      E[log1p Y] = P(Y>0) * E[log1p Y | Y>0]

  THE PRODUCT ITSELF. Multiplying is a hard-coded combination rule: a 10% error in clicks
  becomes a 10% error in Y. That is only right if the factors are independent, and they are not
  — heavy clickers convert at a lower value per click, so the product over-disperses. Nothing
  forces us to multiply; the combination can be LEARNED, which lets a noisy factor be
  down-weighted instead of compounded.

Parts, each a LightGBM on the same 165 features, each trained where it is defined:
    gate     P(Y>0)                              all rows
    k_pos    E[log1p K | Y>0]                    positive rows
    vpc      E[log1p Y - log1p K | Y>0]          positive rows
    conv     E[log1p(ord) - log1p(cart) | Y>0]   positive rows
    aov      E[log1p Y - log1p(ord) | Y>0]       positive rows

Combinations:
    direct        plain model on log1p Y                       <- the control that must be beaten
    prod_median   r5's version, personal median vpc            <- the published failure
    prod_model    gate * (k_pos + vpc)
    triple        gate * (k_pos + conv + aov)
    offset        model on log1p Y with init_score = k_pos      clicks as an offset
    learned       second stage on [parts | top features]        the combination is fitted

`learned` needs the parts to be OUT OF FOLD or it will trust them far more than it should: the
parts are fitted on the same rows the second stage trains on. Leave-one-anchor-out gives every
training anchor parts produced without it, and the evaluation anchor parts from the full fit.
"""
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = (42, 101)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True)
G = panel("gmv")
ORD = panel("ord").astype(np.float32)
CART = panel("cart").astype(np.float32)
VIS = panel("visit").astype(np.float32)

CLICK = {"carts": CART, "visits": VIS}

def win30(M, a):
    return M[:, a + 1:a + 31].sum(1)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
Y = {a: target(a) for a in anchors}
print(f"признаков {X[SEL_EV].shape[1]}, якорей {len(anchors)}")

def parts_for(fit_anchors, out_anchor, click, seed=42):
    """fit every part on fit_anchors, emit them at out_anchor"""
    Xtr = np.concatenate([X[a] for a in fit_anchors], 0)
    y = np.concatenate([Y[a] for a in fit_anchors])
    K = np.concatenate([win30(click, a) for a in fit_anchors])
    O = np.concatenate([win30(ORD, a) for a in fit_anchors])
    C = np.concatenate([win30(CART, a) for a in fit_anchors])
    pos = y > 0
    Xe = X[out_anchor]
    out = {}
    g = lgb.LGBMClassifier(**dict(LGB, objective="binary", random_state=seed)).fit(
        Xtr, pos.astype(int))
    out["gate"] = g.predict_proba(Xe)[:, 1]
    def reg(t, mask):
        m = lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=seed))
        m.fit(Xtr[mask], t[mask])
        return m.predict(Xe)
    lk, ly = np.log1p(K), np.log1p(y)
    out["k_pos"] = reg(lk, pos)
    out["vpc"] = reg(ly - lk, pos)
    out["conv"] = reg(np.log1p(O) - np.log1p(C), pos)
    out["aov"] = reg(ly - np.log1p(O), pos)
    del Xtr
    return out

def vpc_median(a, click, k=6):
    bl = [(max(0, a - 30 * (i + 1) + 1), a - 30 * i) for i in range(k)]
    gm = np.stack([G[:, s:e + 1].sum(1) for s, e in bl], 1)
    cm = np.stack([click[:, s:e + 1].sum(1) for s, e in bl], 1)
    ok = (cm > 0) & (gm > 0)
    lv = np.where(ok, np.log1p(gm) - np.log1p(cm), np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(lv, 1)
    n = ok.sum(1)
    pop = float(np.nanmedian(lv[ok])) if ok.any() else 0.0
    return np.where(n > 0, (np.nan_to_num(med) * n + pop * 2) / (n + 2), pop)

yev = Y[SEL_EV]
lyev = np.log1p(yev)
Xtr_all = np.concatenate([X[a] for a in SEL_TR], 0)
ytr_all = np.concatenate([Y[a] for a in SEL_TR])
ltr_all = np.log1p(ytr_all)

direct = np.mean([lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=s))
                  .fit(Xtr_all, ltr_all).predict(X[SEL_EV]) for s in SEEDS], 0)
print(f"\nконтроль — прямая модель на log1p Y: {rmsle(yev, np.expm1(direct)):.5f}\n")

for cname, click in CLICK.items():
    print(f"=== клик = {cname} ===", flush=True)
    t0 = time.time()
    P = parts_for(SEL_TR, SEL_EV, click)
    lk_ev = np.log1p(win30(click, SEL_EV))
    print(f"  части (на положительных): RMSLE(k_pos) "
          f"{np.sqrt(np.mean((P['k_pos'][yev>0]-lk_ev[yev>0])**2)):.5f}")
    res = {}
    res["prod_median"] = P["gate"] * np.clip(P["k_pos"] + vpc_median(SEL_EV, click), 0, 13)
    res["prod_model"] = P["gate"] * np.clip(P["k_pos"] + P["vpc"], 0, 13)
    res["triple"] = P["gate"] * np.clip(P["k_pos"] + P["conv"] + P["aov"], 0, 13)
    m = lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=42))
    m.fit(Xtr_all, ltr_all - np.concatenate(
        [parts_for([b for b in SEL_TR if b != a], a, click)["k_pos"] for a in SEL_TR]))
    res["offset"] = m.predict(X[SEL_EV]) + P["k_pos"]

    # learned: мета-признаки вне выборки через leave-one-anchor-out
    keys = ["gate", "k_pos", "vpc", "conv", "aov"]
    Mtr = np.concatenate([np.stack([parts_for([b for b in SEL_TR if b != a], a, click)[k]
                                    for k in keys], 1) for a in SEL_TR], 0)
    Mev = np.stack([P[k] for k in keys], 1)
    top = np.argsort(-lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=42))
                     .fit(Xtr_all, ltr_all).booster_.feature_importance("gain"))[:30]
    Ztr = np.concatenate([Mtr, Xtr_all[:, top]], 1)
    Zev = np.concatenate([Mev, X[SEL_EV][:, top]], 1)
    res["learned"] = np.mean([lgb.LGBMRegressor(**dict(LGB, objective="regression",
                                                       random_state=s)).fit(Ztr, ltr_all)
                              .predict(Zev) for s in SEEDS], 0)
    del Ztr, Mtr

    print(f"  {'комбинация':>14} {'RMSLE':>9} {'к контролю':>11} {'corr с контролем':>17}")
    print(f"  {'direct':>14} {rmsle(yev, np.expm1(direct)):9.5f} {0.0:+11.5f} {1.0:17.5f}")
    for k, v in res.items():
        r = rmsle(yev, np.expm1(np.clip(v, 0, 13)))
        print(f"  {k:>14} {r:9.5f} {r-rmsle(yev,np.expm1(direct)):+11.5f} "
              f"{np.corrcoef(np.clip(v,0,13), direct)[0,1]:17.5f}")
    print(f"  ({time.time()-t0:.0f}s)\n", flush=True)
print("done")
