"""Random Forest and Extremely Randomized Trees — the only tree family that is not boosting.

Every tree model in the stack so far fits residuals: LightGBM, CatBoost, XGBoost, DART. Bagged
forests do the opposite — they grow deep, low-bias, high-variance trees on perturbed views and
average them. Same feature store, genuinely different inductive bias, so their errors have a
reason to land somewhere else. That, not their own score, is what the stack pays for: the
admission rule is corr(L, new) > corr(L, stack) * g, and everything added since stack9 has sat
at g >= 0.98.

The variations are ordered along the axis that actually matters here — how much randomisation
is injected, i.e. how decorrelated the trees are from each other and hence from everything else:

  rf_full     RF, max_features=1.0        the regression default; nearly plain bagging
  rf_03       RF, max_features=0.3        feature subsampling, the usual RF recipe
  rf_sqrt     RF, max_features='sqrt'     aggressive subsampling, classic classification recipe
  rf_leaf100  RF, 0.3, min_samples_leaf=100   shallower, closer to the boosting regime
  et_full     ET, max_features=1.0        random split thresholds, all features considered
  et_03       ET, max_features=0.3
  et_sqrt     ET, max_features='sqrt'
  et_mf1      ET, max_features=1          ONE random feature per split — the extreme end of
                                          randomisation, almost a random-projection ensemble
  et_2stage   ET classifier x ET regressor on positives, per the exact identity
                                          E[log1p y] = P(y>0) * E[log1p y | y>0]

Memory: 1.25M rows x 165 float32 is 825 MB, and a forest of 150 deep trees on that many rows
holds a comparable amount again. min_samples_leaf is kept at 20 or above for that reason, and
RF uses max_samples=0.6 so each tree sees a smaller bootstrap.

Reported for every arm: its own RMSLE, and its correlation with the production LightGBM. A weak
score with a low correlation is worth more here than a good score that duplicates the cluster.
"""
import sys
import time

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              ExtraTreesClassifier)

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
NJ = 11
NT = 150

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0).astype(np.float32)
ytr = np.concatenate([target(a) for a in SEL_TR])
ltr = np.log1p(ytr).astype(np.float32)
Xev = X[SEL_EV].astype(np.float32)
yev = target(SEL_EV)
print(f"обучение {Xtr.shape}, оценка {Xev.shape}, {Xtr.nbytes/1e9:.2f} ГБ признаков")

base = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=63,
                         min_child_samples=100, max_bin=63, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=5.0, n_jobs=NJ, verbose=-1,
                         random_state=42, force_col_wise=True,
                         objective="regression").fit(Xtr, ltr).predict(Xev)
print(f"продакшн-LightGBM: {rmsle(yev, np.expm1(base)):.5f}\n")

ARMS = [
    ("rf_full",    RandomForestRegressor, dict(max_features=1.0, min_samples_leaf=20,
                                               max_samples=0.6, bootstrap=True)),
    ("rf_03",      RandomForestRegressor, dict(max_features=0.3, min_samples_leaf=20,
                                               max_samples=0.6, bootstrap=True)),
    ("rf_sqrt",    RandomForestRegressor, dict(max_features="sqrt", min_samples_leaf=20,
                                               max_samples=0.6, bootstrap=True)),
    ("rf_leaf100", RandomForestRegressor, dict(max_features=0.3, min_samples_leaf=100,
                                               max_samples=0.6, bootstrap=True)),
    ("et_full",    ExtraTreesRegressor,   dict(max_features=1.0, min_samples_leaf=20,
                                               bootstrap=False)),
    ("et_03",      ExtraTreesRegressor,   dict(max_features=0.3, min_samples_leaf=20,
                                               bootstrap=False)),
    ("et_sqrt",    ExtraTreesRegressor,   dict(max_features="sqrt", min_samples_leaf=20,
                                               bootstrap=False)),
    ("et_mf1",     ExtraTreesRegressor,   dict(max_features=1, min_samples_leaf=20,
                                               bootstrap=False)),
]

print(f"{'рука':>11} {'RMSLE':>9} {'к LightGBM':>11} {'g':>8} {'листьев/дерево':>15} {'сек':>6}")
RES = {}
for name, cls, kw in ARMS:
    t0 = time.time()
    try:
        m = cls(n_estimators=NT, n_jobs=NJ, random_state=42, **kw).fit(Xtr, ltr)
        p = m.predict(Xev)
        leaves = int(np.mean([t.get_n_leaves() for t in m.estimators_[:10]]))
        del m
    except Exception as e:
        print(f"{name:>11}  ОШИБКА: {type(e).__name__}: {str(e)[:60]}")
        continue
    r = rmsle(yev, np.expm1(np.clip(p, 0, 13)))
    g = float(np.corrcoef(np.clip(p, 0, 13), base)[0, 1])
    RES[name] = p
    print(f"{name:>11} {r:9.5f} {r-rmsle(yev,np.expm1(base)):+11.5f} {g:8.5f} "
          f"{leaves:15d} {time.time()-t0:6.0f}", flush=True)

# двухстадийная версия по точному тождеству
t0 = time.time()
pos = ytr > 0
gate = ExtraTreesClassifier(n_estimators=NT, max_features=0.3, min_samples_leaf=20,
                            n_jobs=NJ, random_state=42).fit(Xtr, pos)
reg = ExtraTreesRegressor(n_estimators=NT, max_features=0.3, min_samples_leaf=20,
                          bootstrap=False, n_jobs=NJ, random_state=42).fit(Xtr[pos], ltr[pos])
p = gate.predict_proba(Xev)[:, 1] * np.clip(reg.predict(Xev), 0, 13)
del gate, reg
r = rmsle(yev, np.expm1(np.clip(p, 0, 13)))
g = float(np.corrcoef(np.clip(p, 0, 13), base)[0, 1])
RES["et_2stage"] = p
print(f"{'et_2stage':>11} {r:9.5f} {r-rmsle(yev,np.expm1(base)):+11.5f} {g:8.5f} "
      f"{'—':>15} {time.time()-t0:6.0f}")

print(f"\nвзаимные корреляции лесов (насколько вариации отличаются друг от друга):")
ks = list(RES)
print("            " + " ".join(f"{k[:8]:>8}" for k in ks))
for a_ in ks:
    print(f"  {a_:>10} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:8.4f}" for b in ks))

print(f"\nпорог допуска в стек: при corr(L, стек) = 0,70333 компонента полезна,")
print(f"если её g к стеку ниже её собственной corr(L)/0,70333.")
print(f"Здесь g меряется к LightGBM, а не к стеку — это верхняя оценка снизу:")
print(f"стек коррелирован со всем сильнее одиночной модели, так что настоящий g будет выше.")
print("done")
