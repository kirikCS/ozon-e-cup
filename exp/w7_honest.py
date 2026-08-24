"""Честная оценка набора: отбор по одной половине, замер по другой.

Жадный отбор в w2 выбирает лучшего из 40 кандидатов на каждом шаге и оценивает по СРЕДНЕМУ двух
половин — то есть выбирает по тем же данным, по которым отчитывается. Это проклятие победителя:
при 40 кандидатах и вкладах порядка 1e-04 часть выигрыша может быть подгонкой под шум выбора,
а контрольный перемешанный столбец такую подгонку не ловит — он меряет цену СТОЛБЦА, а не цену
ВЫБОРА.

Здесь отбор идёт строго по направлению A→B, а отчёт — по B→A, которое в отборе не участвовало.
Разрыв между двумя числами и есть величина проклятия победителя.

Если честный выигрыш сохраняется — набор стоит собирать. Если схлопывается — жадный отбор
показывал шум, и брать нечего.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

import re as _re
# ЛЮБОЙ вектор вида <буква><цифры>_... — кандидат. Перечисление префиксов подводило четырежды:
# забытый v11 не совпал с "v1_" и уехал В БАЗИС, изменив все вклады. Регулярка не забывается.
CANDRX = _re.compile(r"^[suvwxy]\d+_")

LEAK_RMSLE = 1.70

def leaky(name, score):
    return ("fin" in name) or (score < LEAK_RMSLE)

PREF = tuple(f"{k}_" for k in ("s6", "s7", "s8", "u1", "u2", "u3", "u4", "u5", "u6",
                               "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"))
NSTEP = int(__import__("os").environ.get("NSTEP", 8))
y = target(348); ly = np.log1p(y)
BASE, CAND, DROP = {}, {}, []
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (250000,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= 1.80:
        continue
    if leaky(p.stem, sc):
        DROP.append(p.stem); continue
    (CAND if CANDRX.match(p.stem) else BASE)[p.stem] = v - v.mean()
print(f"отброшено по утечке {len(DROP)}: {DROP}")
print(f"в базисе {len(BASE)}, кандидатов {len(CAND)}", flush=True)

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
SEL = (half, ~half)      # на этом ОТБИРАЕМ
REP = (~half, half)      # на этом ОТЧИТЫВАЕМСЯ, в отборе не участвует
cols = list(BASE.values())
cur_sel = fit_eval(cols, *SEL)
cur_rep = fit_eval(cols, *REP)
print(f"\nбаза: отбор {cur_sel:.6f}, отчёт {cur_rep:.6f}\n", flush=True)
taken = []
for step in range(NSTEP):
    t0 = time.time()
    best = None
    for name, v in CAND.items():
        if name in taken:
            continue
        d = fit_eval(cols + [v], *SEL) - cur_sel
        if best is None or d < best[0]:
            best = (d, name, v)
    d, name, v = best
    new_rep = fit_eval(cols + [v], *REP)
    drep = new_rep - cur_rep
    print(f"шаг {step+1}: {name:>22} отбор {d:+.2e}   ЧЕСТНЫЙ {drep:+.2e}   "
          f"({time.time()-t0:.0f}s)", flush=True)
    if d >= 0:
        print("  на отборочной половине выигрыша нет — стоп")
        break
    taken.append(name); cols.append(v)
    cur_sel += d; cur_rep = new_rep
print(f"\nвзято {len(taken)}: {taken}")
print(f"итог на отборочной половине: {cur_sel:.6f}")
print(f"итог на ЧЕСТНОЙ половине:    {cur_rep:.6f}")
print("\nразрыв между ними — величина проклятия победителя")
print("\ndone")
