"""Проверка замены компоненты на стенде: тот же сдвиг, что пойдёт в сабмит.

Сборка x12 добавляет к прогнозу w_j*(c_j' - c_j), где w_j — вес компоненты из риджа. Здесь ровно
та же операция применяется к прогнозу СТЕНДА на якоре 348, и смотрится скор на половине B->A,
на которой веса не подбирались.

Зачем это до отправки. Сдвиг имеет размах 0,0338 в логарифме; если направление пустое, цена по
арифметике зонда eps^2/(2*R0) около 3,4e-04, тогда как наш отрыв от второго места 1,63e-05.
Ставка с таким плечом требует локального свидетельства, а не надежды.

Перебирается и множитель силы: замена на полной силе может быть хуже, чем на половинной, потому
что цена растёт квадратично по множителю, а выигрыш линейно.
"""
import sys, os
sys.path.insert(0, "code")
import re as _re
import numpy as np
from pathlib import Path
from common import target, rmsle, NU, OUT

CANDRX = _re.compile(r"^[suvwx]\d+_")
LEAK = 1.70
y = target(348); ly = np.log1p(y)
BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (NU,) or not np.isfinite(v).all():
        continue
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if s >= 1.80 or "fin" in p.stem or s < LEAK or CANDRX.match(p.stem):
        continue
    BASE[p.stem] = v - v.mean()
names, cols = list(BASE), list(BASE.values())
print(f"в базисе {len(BASE)}", flush=True)

OLD, NEW = os.environ.get("PAIR", "x16_ev_old,x16_ev_new").split(",")
o = np.load(OUT / f"{OLD}_sel.npy").astype(np.float64)
n = np.load(OUT / f"{NEW}_sel.npy").astype(np.float64)
d = (n - n.mean()) - (o - o.mean())
cols.append(o - o.mean()); names.append(OLD)

rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
fm, em = half, ~half          # веса на A, отчёт на B
M = np.stack(cols, 1)
A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
lam = 1e-4
G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
w = np.linalg.solve(G, A.T @ ly[fm])
wj = float(w[names.index(OLD)])
pred = M @ w[:-1] + w[-1]
b = rmsle(y[em], np.expm1(np.clip(pred[em], 0, 13)))
print(f"вес компоненты w_j = {wj:+.5f}")
print(f"стенд без замены: {b:.6f}\n")
print(f"{'множитель':>10} {'скор':>10} {'сдвиг':>11}")
for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    r = rmsle(y[em], np.expm1(np.clip((pred + k * wj * d)[em], 0, 13)))
    print(f"{k:>10.2f} {r:10.6f} {r-b:+11.2e}")
print("\nотрицательный сдвиг = замена улучшает стенд на невиданной половине")
print("\ndone")
