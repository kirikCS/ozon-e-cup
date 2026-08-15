"""What is one click of this client worth, and can next month be predicted from it?

Y = clicks * value_per_click. In log space that is a SUM, not a sum of logs of a sum — which
makes this the one decomposition in this problem that fits RMSLE natively. Every other split
tried so far (daily values, channel totals) had to fight Jensen; this one does not, because
log of a product is exactly the sum of logs.

    log1p(Y)  ~  log(clicks) + log(value_per_click)

and the two factors have very different statistics: clicks are dense (80.7% of active days
carry a search, 3.2 searches per active day) and therefore far more predictable than GMV, while
value-per-click is close to a personal constant — which is exactly what the median over months
estimates robustly.

Four click definitions, because "click" is not defined in the data and the right proxy is an
empirical question: searches, carts, visit-days and engaged-days.

Three arms:
  features    per-client median value-per-click, its dispersion, its trend and the count of
              months it rests on — added to the production LightGBM
  product     the standalone multiplicative model: predict log(clicks) with a tree, add the
              shrunk median log value-per-click, exponentiate. No Jensen correction needed
  control     the same tree predicting log(clicks) but multiplied by the POPULATION median
              instead of the personal one — isolates what the personal constant adds
"""
import sys

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = (42, 101, 777)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")

G = panel("gmv")
CLICKS = {"searches": panel("srch").astype(np.float32),
          "carts": panel("cart").astype(np.float32),
          "visits": panel("visit").astype(np.float32),
          "engaged": panel("engaged").astype(np.float32)}

def blocks_back(a, k=6):
    return [(max(0, a - 30 * (i + 1) + 1), a - 30 * i) for i in range(k)]

def vpc_stats(a, C, k=6):
    """per-month log value-per-click, then the median over months — history only"""
    bl = blocks_back(a, k)
    gm = np.stack([G[:, s:e + 1].sum(1) for s, e in bl], 1)
    cm = np.stack([C[:, s:e + 1].sum(1) for s, e in bl], 1)
    ok = (cm > 0) & (gm > 0)
    lv = np.where(ok, np.log1p(gm) - np.log1p(cm), np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(lv, 1)
        disp = np.nanstd(lv, 1)
    n = ok.sum(1)
    pop = float(np.nanmedian(lv[ok])) if ok.any() else 0.0
    # усадка к популяционной медиане: вес приора — 2 месяца
    med_s = np.where(n > 0, (np.nan_to_num(med) * n + pop * 2) / (n + 2), pop)
    recent = np.where(ok[:, 0], np.nan_to_num(lv[:, 0]), med_s)
    return med_s, np.nan_to_num(disp), n.astype(np.float32), recent - med_s, pop

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
ytr_raw = np.concatenate([target(a) for a in SEL_TR])
ytr = np.log1p(ytr_raw)
yev = target(SEL_EV)

print(f"=== цена клика, оценка {ds(SEL_EV)} ===")
print(f"{'определение клика':>18} {'клиентов с оценкой':>19} {'медиана log(GMV/клик)':>22} "
      f"{'sd между клиентами':>19} {'sd внутри клиента':>18}")
for nm, C in CLICKS.items():
    med, disp, n, tr, pop = vpc_stats(SEL_EV, C)
    have = n > 0
    print(f"{nm:>18} {int(have.sum()):19d} {pop:22.4f} {med[have].std():19.4f} "
          f"{disp[have].mean():18.4f}")

print("\n--- 1. как признаки поверх продакшн-LightGBM ---")
def cv_feats(a):
    cols = []
    for nm, C in CLICKS.items():
        med, disp, n, tr, _ = vpc_stats(a, C)
        cols += [med, disp, n, tr]
    return np.stack(cols, 1).astype(np.float32)

for tag, use in (("база", False), ("+цена клика", True)):
    Xtr = np.concatenate([np.concatenate([X[a], cv_feats(a)], 1) if use else X[a]
                          for a in SEL_TR], 0)
    Xev = np.concatenate([X[SEL_EV], cv_feats(SEL_EV)], 1) if use else X[SEL_EV]
    s = [rmsle(yev, np.expm1(lgb.LGBMRegressor(**dict(LGB, random_state=sd))
                             .fit(Xtr, ytr).predict(Xev))) for sd in SEEDS]
    print(f"  {tag:16s} {np.mean(s):.5f}   сиды " + ", ".join(f"{v:.5f}" for v in s))
    del Xtr

print("\n--- 2. мультипликативная модель: клики x цена клика ---")
print(f"  {'клик':>10} {'RMSLE(клики)':>13} {'произведение':>13} {'популяц. константа':>19}")
Xtr_all = np.concatenate([X[a] for a in SEL_TR], 0)
for nm, C in CLICKS.items():
    ctr = np.concatenate([np.log1p(C[:, a + 1:a + 31].sum(1)) for a in SEL_TR])
    cev = np.log1p(C[:, SEL_EV + 1:SEL_EV + 31].sum(1))
    ps = []
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**dict(LGB, random_state=sd)).fit(Xtr_all, ctr)
        ps.append(m.predict(X[SEL_EV]))
    ch = np.mean(ps, 0)
    rc = float(np.sqrt(np.mean((ch - cev) ** 2)))
    med, _, n, _, pop = vpc_stats(SEL_EV, C)
    r_pers = rmsle(yev, np.expm1(np.clip(ch + med, 0, 13)))
    r_pop = rmsle(yev, np.expm1(np.clip(ch + pop, 0, 13)))
    print(f"  {nm:>10} {rc:13.5f} {r_pers:13.5f} {r_pop:19.5f}")
print("\nчитать так: 'произведение' должно бить 'популяционную константу' — иначе личная цена клика ничего не несёт")
print("done")
