"""Does capping the history window help, now that a structural break is dated?

The channel EDA found one: in April 2025 the catalog average order value jumps 37.42 -> 44.94 in
a single month and its share of orders falls 0.0800 -> 0.0575, while search moves smoothly
through the same months (35.42 -> 35.27 -> 34.25). That is a product change, not a season. Any
feature averaging over more than about eleven months therefore mixes two regimes, and 24 of the
165 columns do exactly that — the _365 and _all aggregates.

Two reasons to expect nothing, stated before the measurement so the result cannot be rationalised
afterwards:

  the break is confined to a channel worth 7.4% of GMV, and the catalog features built to carry
  it were already measured at zero (1.74243 -> 1.74247);

  RFE cut the store from 165 columns to 30 without changing quality, so the long windows are
  almost certainly reconstructible from the short ones.

Arms, all on the same rows and seeds — only the column set differs:

  full        all 165                              control
  no_365      drop the 24 _365 / _all columns      windows capped at 180 days
  cap_90      drop _180 as well                    capped at 90 days, past the break entirely
  only_long   keep ONLY the 24 long columns        the mirror image: if the long windows were
                                                   pure contamination, this arm should be much
                                                   worse than its column count suggests
"""
import sys
import re
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import target, rmsle, build_features, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = (42, 101, 777)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")

X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
names = np.array(names)
is_long = np.array([bool(re.search(r"_(365|all)(_|$)", n) or n.endswith("_365")
                         or n.endswith("_all")) for n in names])
is_180 = np.array([("_180" in n) for n in names])
print(f"признаков {len(names)}: длинных (365/all) {is_long.sum()}, со 180 {is_180.sum()}")

ARMS = {
    "full": np.ones(len(names), bool),
    "no_365": ~is_long,
    "cap_90": ~(is_long | is_180),
    "only_long": is_long,
}

ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
yev = target(SEL_EV)
print(f"\n{'рука':>11} {'колонок':>8} {'RMSLE':>9} {'к full':>9}   сиды")
base = None
P = {}
for tag, m in ARMS.items():
    Xtr = np.concatenate([X[a][:, m] for a in SEL_TR], 0)
    Xev = X[SEL_EV][:, m]
    s = [rmsle(yev, np.expm1(lgb.LGBMRegressor(**dict(LGB, random_state=sd))
                             .fit(Xtr, ytr).predict(Xev))) for sd in SEEDS]
    del Xtr
    P[tag] = np.mean(s)
    if base is None:
        base = P[tag]
    print(f"{tag:>11} {int(m.sum()):8d} {P[tag]:9.5f} {P[tag]-base:+9.5f}   "
          + ", ".join(f"{v:.5f}" for v in s), flush=True)

print(f"\nполоса шума этого экрана ~ ±0.0004 (разброс по сидам внутри руки)")
print("читать так: если no_365 и cap_90 не лучше full, длинные окна не мешают —")
print("апрельский перелом сидит в канале, который весит 7,4% и уже измерен нулём")
print("done")
