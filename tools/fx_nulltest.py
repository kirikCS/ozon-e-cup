"""НУЛЕВОЙ тест: сколько выигрыша на public даёт чистая подгонка под шум.

Первая симуляция (fx_pubpriv.py) дала перенос 0.57-0.66 и 100% улучшений на
привате. Но там опорной точкой были равные веса - заведомо неоптимальные, поэтому
почти весь выигрыш был НАСТОЯЩИМ и, естественно, переносился. У нас опора - рекорд,
уже сильно оптимизированный стек. Ситуация другая.

Решающий тест. Берём опору w_full = оптимум по ВСЕМ 250 000 (лучшее, что вообще
достижимо - настоящего выигрыша от неё не осталось). Оттуда оптимизируем ТОЛЬКО по
публичным 50k в шаре радиуса D. Весь найденный публичный выигрыш - чистая подгонка
под шум публичной подвыборки. Смотрим:
  * сколько public-выигрыша даёт чистый шум на каждом радиусе
  * сколько при этом ТЕРЯЕТСЯ на привате

Затем сравниваем: наш кандидат имеет ||z||_C = 0.0274 и публичный выигрыш +2.3e-4.
Если нулевой тест на том же радиусе даёт БОЛЬШЕ - наш выигрыш неотличим от шума.

Usage: python -u code/fx_nulltest.py > model_out/nulltest.log 2>&1
"""
import sys
import glob
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, target

N_SPLIT = 40
N_PUB = 50_000
RNG = np.random.default_rng(4242)

y = np.asarray(target(VAL_ANCHOR), dtype=np.float64)
ly = np.log1p(y)
n = len(y)
files = sorted(set(glob.glob("model_out/*_VAL_*.npy")) | set(glob.glob("model_out/*_VAL.npy")))
P, names = [], []
for f in files:
    a = np.load(f).astype(np.float64)
    if a.shape != (n,):
        continue
    q = a if a.mean() < 10 else np.log1p(a)
    P.append(q - q.mean() + ly.mean()); names.append(Path(f).stem)
M = np.stack(P); NB = len(names)
C_all = np.cov(M)
print(f"{n} клиентов, {NB} векторов")


def score(w, idx):
    q = w @ M[:, idx]
    q = q + (ly[idx].mean() - q.mean())
    return float(np.sqrt(np.mean((q - ly[idx])**2)))


# --- опора: оптимум по ВСЕМ 250k -------------------------------------------------
k_all = np.array([np.cov(M[i], ly)[0, 1] for i in range(NB)])
w_full = np.linalg.solve(C_all + 1e-9*np.eye(NB), k_all)
print(f"опора w_full (оптимум по всем 250k): RMSLE = {score(w_full, slice(None)):.6f}")
print(f"  для сравнения, равные веса        : {score(np.full(NB,1/NB), slice(None)):.6f}")
print(f"  сумма весов {w_full.sum():.4f}  max|w| {np.abs(w_full).max():.3f}")

print()
print("=" * 80)
print("НУЛЕВОЙ ТЕСТ: оптимизация по 50k из точки, где настоящего выигрыша НЕТ")
print("=" * 80)
print(f"{'D':>7} {'public выигрыш':>16} {'private ИЗМЕНЕНИЕ':>19} {'доля улучш.priv':>16}")
null_pub = {}
for D in [0.005, 0.010, 0.020, 0.027, 0.035, 0.050]:
    gp, gv = [], []
    for s in range(N_SPLIT):
        perm = RNG.permutation(n)
        pub, priv = perm[:N_PUB], perm[N_PUB:]
        Mp = M[:, pub]; yp = ly[pub]; Cp = np.cov(Mp)
        dist = lambda z: float(np.sqrt(max(z @ Cp @ z, 0)))

        def m2(w):
            q = w @ Mp
            return float(np.mean((q + (yp.mean() - q.mean()) - yp)**2))

        r = minimize(lambda z: m2(w_full + z), np.zeros(NB), method="SLSQP",
                     constraints=[{"type": "ineq", "fun": lambda z: D - dist(z)}],
                     options={"maxiter": 200, "ftol": 1e-14})
        w = w_full + r.x
        gp.append(score(w_full, pub) - score(w, pub))
        gv.append(score(w_full, priv) - score(w, priv))
    gp, gv = np.array(gp), np.array(gv)
    null_pub[D] = gp.mean()
    print(f"{D:7.3f} {gp.mean():+16.6f} {gv.mean():+19.6f} {100*np.mean(gv>0):15.0f}%")
    print(f"{'':7} {'±':>16}{gp.std():.6f} {'±':>13}{gv.std():.6f}")

print()
print("public выигрыш здесь - ЧИСТЫЙ ШУМ: настоящего выигрыша из w_full не бывает.")
print("private изменение показывает, чем за него платят.")

print()
print("=" * 80)
print("СРАВНЕНИЕ С НАШИМ КАНДИДАТОМ")
print("=" * 80)
OUR_D, OUR_GAIN = 0.0274, 0.000228
lo = max(d for d in null_pub if d <= OUR_D)
hi = min(d for d in null_pub if d >= OUR_D)
nz = (null_pub[lo] + null_pub[hi]) / 2 if lo != hi else null_pub[lo]
print(f"SAFE_beat_record : ||z||_C = {OUR_D:.4f}, публичный выигрыш +{OUR_GAIN:.6f}")
print(f"чистый шум на том же радиусе даёт публичный выигрыш +{nz:.6f}")
print(f"отношение сигнал/шум = {OUR_GAIN/nz:.2f}")
if OUR_GAIN < nz:
    print("=> наш публичный выигрыш МЕНЬШЕ шумового. Неотличим от подгонки.")
else:
    print("=> наш публичный выигрыш БОЛЬШЕ шумового. Часть его настоящая:")
    print(f"   грубая оценка настоящей части +{OUR_GAIN-nz:.6f}")
print()
print("ВАЖНО про размерность: здесь 20 векторов и 50k, у нас 26 векторов и 50k.")
print("Шумовой выигрыш растёт примерно как sqrt(d), поправка x"
      f"{(26/NB)**0.5:.2f} -> +{nz*(26/NB)**0.5:.6f}")
