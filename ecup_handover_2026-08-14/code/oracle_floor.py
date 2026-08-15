"""How low can RMSLE physically go on this data? Measured, not argued.

Every number here is deliberately cheating: the model is trained on the very rows it is then
scored on, at the validation anchor. That is not a model, it is an upper bound on what any
model built from these features could ever achieve — an honest model must do worse, because
it cannot see the answers.

Three levels of cheating, each stronger than the last:

  1  the production LightGBM, fitted and scored in-sample
  2  a deliberately over-parameterised one — 1023 leaves, no regularisation, 3000 trees
  3  the pure-noise floor: a model that knows E[log1p y] exactly inside each cell of a
     coarse feature grid, again in-sample

If even level 2 cannot approach a target, that target is unreachable by any means, and a
model that reports it has broken its validation rather than solved the problem.
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, NU, target, rmsle, build_features, ds

A = VAL_ANCHOR
X, names = build_features([A])
y = target(A)
ly = np.log1p(y)
Xa = X[A]
print(f"anchor {ds(A)}, {Xa.shape[0]} users, {Xa.shape[1]} features")
print(f"Var(log1p y) = {ly.var():.4f}   RMSLE of the constant mean = {ly.std():.5f}\n")

print(f"{'что это':<58}{'RMSLE':>9}")
print(f"{'наивный prev30 (бейзлайн организаторов)':<58}{2.19506:>9.5f}")
print(f"{'текущий результат на лидерборде (stack6)':<58}{1.64966:>9.5f}")

t0 = time.time()
m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=63,
                      min_child_samples=100, max_bin=63, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=5.0, n_jobs=23, verbose=-1,
                      random_state=42, force_col_wise=True, objective="regression")
m.fit(Xa, ly)
s1 = rmsle(y, np.expm1(m.predict(Xa)))
print(f"{'1. рабочая LightGBM, обучена и замерена на одних строках':<58}{s1:>9.5f}   ({time.time()-t0:.0f}s)")

t0 = time.time()
m2 = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.1, num_leaves=1023,
                       min_child_samples=2, min_child_weight=0.0, max_bin=255,
                       subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0,
                       n_jobs=23, verbose=-1, random_state=42, force_col_wise=True,
                       objective="regression")
m2.fit(Xa, ly)
s2 = rmsle(y, np.expm1(m2.predict(Xa)))
print(f"{'2. то же, но 1023 листа / 3000 деревьев / без регуляризации':<58}{s2:>9.5f}   ({time.time()-t0:.0f}s)")

# level 3: exact in-sample cell means over a coarse grid of the strongest predictors
cols = [names.index(n) for n in ["buyday_all", "gmv_90", "buyday_30", "rec_buy", "gmv_30"]]
q = np.stack([np.searchsorted(np.percentile(Xa[:, c], np.arange(5, 100, 5)), Xa[:, c])
              for c in cols], 1)
key = np.ravel_multi_index(tuple(q.T), (21,) * len(cols))
order = np.argsort(key, kind="stable")
ks, lys = key[order], ly[order]
bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
pred = np.empty(NU)
for i in range(len(bounds) - 1):
    a, b = bounds[i], bounds[i + 1]
    pred[order[a:b]] = lys[a:b].mean()
s3 = rmsle(y, np.expm1(pred))
ncell = len(bounds) - 1
print(f"{'3. точное среднее log1p внутри ячейки по 5 признакам':<58}{s3:>9.5f}   "
      f"({ncell} ячеек, {NU/ncell:.1f} чел/ячейка)")

print(f"\nуровень 2 выродился в запоминание: 1023 листа при min_child_samples=2 на 250k строк")
print(f"просто заучивают ответ, поэтому 0.0018 — не оценка потолка, а доказательство, что")
print(f"интерполяцией можно получить любое число. Границу задают уровни 1 и 3.")

# Level 3 fits each cell mean on the same points it scores, so it understates the true
# conditional spread by exactly (1 - 1/n). Undo that and what is left is the noise floor:
# how much two customers with the same history still differ.
n = NU / ncell
sigma = s3 / np.sqrt(1 - 1 / n)
floor_share = sigma ** 2 / ly.var()
print(f"\nвнутри ячейки сидит {n:.1f} человек с почти одинаковой историей, и разброс их")
print(f"log1p(GMV) в выборке {s3:.5f}. Поправка на то, что среднее считалось по тем же точкам,")
print(f"даёт условное СКО {sigma:.4f} — это и есть неустранимый шум: {100*floor_share:.1f}% всей дисперсии.")
print(f"\nпотолок для честной модели на этих признаках  ~{sigma:.3f}")
print(f"текущий результат на лидерборде                {1.64966:.3f}")
for t in (1.2, 1.4, 1.6):
    verdict = "достижима" if t > sigma else "НЕДОСТИЖИМА: ниже неустранимого шума"
    print(f"  цель RMSLE {t}: {verdict}")
