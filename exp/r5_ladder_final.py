"""Лестница порогов на ФИНАЛЬНОМ якоре — продакшн-версия для стека.

На SELECT-протоколе (r2) лестница дала RMSLE 1,73556 — лучше нашей свёртки (1,73808) и бустинга
(1,73831), — и вклад в стенд из 55 векторов −5,32e-04 при контроле −2,16e-06. Это крупнейший
вклад за проект: квантильные головы давали −3,95e-4, кликовый зонд −1,96e-4.

Тождество: E[S|x] = sum_m P(S > s_m|x) * delta даёт ровно функционал, оптимальный под RMSLE.
Нулевая масса берётся сама собой — нижняя перекладина это P(Y>0). Каждый классификатор видит
все 250 000 клиентов, меняется только метка.

Здесь обучение на FIN_TR с предсказанием на 408. Цели за 408 не существует (панель 409 дней),
поэтому скора нет — только вектор.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import target, rmsle, build_features, NU, OUT

SEL_TR, SEL_EV = [378, 350, 322, 294, 266], 408   # ФИНАЛЬНЫЙ протокол
DELTA = float(__import__("os").environ.get("DELTA", 0.25))
y = None
print("фичестор ...", flush=True)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
Xev = X[SEL_EV]
TOP = float(np.quantile(ytr, 0.9995))
GRID = np.arange(DELTA, TOP, DELTA)
print(f"  {Xtr.shape}, порогов {len(GRID)}, шаг {DELTA}, верх {TOP:.2f}", flush=True)

P = np.zeros((NU, len(GRID)), np.float32)
t0 = time.time()
for j, s in enumerate(GRID):
    lab = (ytr > s).astype(np.int8)
    fr = lab.mean()
    if fr < 1e-4 or fr > 1 - 1e-4:
        P[:, j] = fr; continue
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.07, num_leaves=63,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    m.fit(Xtr, lab)
    P[:, j] = m.predict_proba(Xev)[:, 1]
    if j % 8 == 0:
        print(f"  порог {s:5.2f}: доля {fr:.4f}, средняя p {P[:,j].mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
np.save(OUT / "r5_ladder_P_fin.npy", P)

from common import N_DAYS, H
assert SEL_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
mean_ro = DELTA * P.sum(1)
Pm = np.maximum.accumulate(P[:, ::-1], 1)[:, ::-1]
med_ro = np.where((Pm > 0.5).any(1), GRID[np.argmin(Pm > 0.5, 1)], 0.0)
np.save(OUT / "r5_ladder_mean_fin.npy", mean_ro.astype(np.float32))
np.save(OUT / "r5_ladder_median_fin.npy", med_ro.astype(np.float32))
print(f"\nинтеграл: средняя log1p {mean_ro.mean():.5f}, sd {mean_ro.std():.5f}")
print(f"медиана:  средняя log1p {med_ro.mean():.5f}, sd {med_ro.std():.5f}")
print(f"corr(интеграл, медиана) = {np.corrcoef(mean_ro, med_ro)[0,1]:.5f}")
print("-> r5_ladder_mean_fin.npy, r5_ladder_median_fin.npy")
print("\ndone")
