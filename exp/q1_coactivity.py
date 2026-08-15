"""Is there a user-user graph in this data at all? Measured, not argued.

There are no explicit edges: the panel is per-user-per-day aggregates, no item ids, no
categories, no shared entities. So any graph would have to be built from behavioural
similarity — and a graph built from features the model already consumes cannot add
information to that model, only smooth it. The teammate already closed the smoothing
version (cell-mean / kNN neighbour targets).

One version survives that argument. If some users are driven by a shared external cause —
a regional promo, a category trend, a household sharing an account — their daily activity
correlates BEYOND the global calendar, and that correlation is not a function of any
per-user aggregate. This is testable with random matrix theory:

  double-center the daily activity matrix (removes both the global calendar and per-user
  level), standardise rows, take the day-day covariance. Under the null of independent
  users, its spectrum obeys Marchenko-Pastur with ratio p/n = 409/250000 = 0.0016, i.e.
  every eigenvalue lands inside [1-2*sqrt(r), 1+2*sqrt(r)] = [0.92, 1.08].

Eigenvalues above that band are real shared factors. If the spectrum is clean, the whole
graph/clustering direction is dead — one measurement, no submissions.

Stage 2 runs only if stage 1 finds factors: project users on them and check whether the
loadings predict the target beyond the existing feature store.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

print("=== 1. спектр остаточной ко-активности ===")
t0 = time.time()
A = (panel("gmv") > 0).astype(np.float32)          # покупка в этот день
print(f"матрица {A.shape}, доля активных ячеек {A.mean():.4f}")

# двойное центрирование: убирает и календарь (среднее по дню), и уровень клиента
row = A.mean(1, keepdims=True)
col = A.mean(0, keepdims=True)
R = A - row - col + A.mean()
sd = R.std(1, keepdims=True)
R /= np.maximum(sd, 1e-6)                          # стандартизация по клиенту
print(f"остатки готовы ({time.time()-t0:.0f}s)")

# day-day ковариация имеет те же нетривиальные с.з., что и user-user, но она 409x409
Cd = (R.T @ R) / NU
ev = np.linalg.eigvalsh(Cd)[::-1]
r = N_DAYS / NU
lo, hi = (1 - np.sqrt(r)) ** 2, (1 + np.sqrt(r)) ** 2
print(f"\nполоса Марченко-Пастура при p/n = {r:.5f}: [{lo:.4f}, {hi:.4f}]")
print(f"топ-15 собственных значений:")
print("  " + " ".join(f"{v:.3f}" for v in ev[:15]))
n_out = int((ev > hi).sum())
print(f"\nвыше полосы: {n_out} из {N_DAYS}   максимум {ev[0]:.4f}  (порог {hi:.4f})")
if ev[0] < hi * 1.05:
    print("ВЫВОД: скрытых общих факторов нет — направление граф/кластеризация закрыто")
else:
    print(f"ВЫВОД: {n_out} факторов сверх шума — переходим к стадии 2")

# ---- стадия 2: несут ли нагрузки на факторы сигнал сверх фичестора
K = max(1, min(n_out, 12))
w, V = np.linalg.eigh(Cd)
F = V[:, ::-1][:, :K]                              # день x фактор
print(f"\n=== 2. нагрузки на {K} факторов как признаки ===")
anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)

import lightgbm as lgb
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")

def load_at(a):
    """loadings computed only on history up to the anchor — no target-window leakage"""
    Ra = R[:, :a + 1]
    return (Ra @ F[:a + 1]) / np.sqrt(a + 1)

ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
yev = target(SEL_EV)
for tag, extra in (("база", False), ("+нагрузки", True)):
    Xtr = np.concatenate([np.concatenate([X[a], load_at(a)], 1) if extra else X[a]
                          for a in SEL_TR], 0)
    Xev = np.concatenate([X[SEL_EV], load_at(SEL_EV)], 1) if extra else X[SEL_EV]
    s = []
    for seed in (42, 101, 777):
        m = lgb.LGBMRegressor(**dict(LGB, random_state=seed)).fit(Xtr, ytr)
        s.append(rmsle(yev, np.expm1(m.predict(Xev))))
    print(f"  {tag:12s} {np.mean(s):.5f}  (сиды {', '.join(f'{v:.5f}' for v in s)})")
    del Xtr
print("\ndone")
