"""Обратный отбор: какие векторы базиса улучшают стенд своим ОТСУТСТВИЕМ.

w8 наткнулся на это случайно: удаление c5_sel улучшает честную половину на -4,46e-05 — больше,
чем добавление лучшей руки календарной оси (-2,42e-05). Значит базис несёт мёртвый груз, и его
вычистка может стоить больше, чем всё, что настроено за ночь.

Механизм понятен: ридж с 71 столбцом при корреляциях 0,99 между ними сильно вырожден. Лишний
почти-дубликат не добавляет информации, но добавляет степень свободы, по которой можно
переобучиться. Регуляризация lam=1e-4 это гасит лишь частично.

Отчёт по половине B->A, отбор по A->B — тот же принцип, что в w7: удалять выбираем по одной
половине, а честную цену смотрим по другой. Без этого обратный отбор просто найдёт шум.

Осторожно с переносом на настоящий стек: здесь базис из 71 вектора, а в стеке отобранный набор
из ~35-40. Вывод «в стенде лишний» не означает «в стеке лишний», но указывает, где искать.
"""
import sys, os, time
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

NSTEP = int(os.environ.get("NSTEP", 8))
LEAK_RMSLE = 1.70

def leaky(name, score):
    return ("fin" in name) or (score < LEAK_RMSLE)

PREF = tuple(f"{k}_" for k in ("s6", "s7", "s8", "u1", "u2", "u3", "u4", "u5", "u6",
                               "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "u7"))
y = target(348); ly = np.log1p(y)
BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (250000,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= 1.80 or leaky(p.stem, sc) or p.stem.startswith(PREF):
        continue
    BASE[p.stem] = v - v.mean()
print(f"в базисе {len(BASE)}", flush=True)

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
SEL, REP = (half, ~half), (~half, half)
names = list(BASE)
cur_sel = fit_eval([BASE[n] for n in names], *SEL)
cur_rep = fit_eval([BASE[n] for n in names], *REP)
print(f"база: отбор {cur_sel:.6f}, отчёт {cur_rep:.6f}\n", flush=True)
dropped = []
for step in range(NSTEP):
    t0 = time.time()
    best = None
    for n in names:
        d = fit_eval([BASE[m] for m in names if m != n], *SEL) - cur_sel
        if best is None or d < best[0]:
            best = (d, n)
    d, n = best
    new_rep = fit_eval([BASE[m] for m in names if m != n], *REP)
    drep = new_rep - cur_rep
    print(f"шаг {step+1}: убрать {n:>22} отбор {d:+.2e}   ЧЕСТНЫЙ {drep:+.2e}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    if d >= 0:
        print("  на отборочной половине удалять больше нечего — стоп")
        break
    dropped.append(n); names.remove(n)
    cur_sel += d; cur_rep = new_rep
print(f"\nубрано {len(dropped)}: {dropped}")
print(f"итог на отборочной половине: {cur_sel:.6f}")
print(f"итог на ЧЕСТНОЙ половине:    {cur_rep:.6f}")
print("\ndone")
