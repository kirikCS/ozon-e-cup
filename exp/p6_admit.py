"""Проходят ли квантильные руки порог допуска в стек — прямая проверка, а не по корреляции.

p5 показал, что квантильные руки ортогональнее к дневной модели, чем генеративные (0,99213 и
0,99241 против 0,99298 и 0,99349), а 0,9921 - нижняя граница полосы «смена архитектуры», где
сидит event_multi: он вошёл в стек с g = 0,99336 и получил наибольший вес из четвёрки, имея
худший локальный скор. Значит по одной корреляции судить нельзя.

Здесь допуск проверяется прямо. Гребневая смесь всех наших векторов на якоре 348 - это стенд
вместо стека. Веса подгоняются на половине клиентов, скор считается на другой, обе стороны.
Кандидат считается полезным, если добавление его в набор улучшает скор на НЕВИДАННОЙ половине.

Контроль обязателен: тот же кандидат, но перемешанный по клиентам. Он несёт ровно один лишний
столбец и никакой информации, поэтому показывает цену самого расширения базиса.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

y = target(348); ly = np.log1p(y)
BASE, CAND = {}, {}
for p in sorted(Path(OUT).glob("*.npy")):
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (250000,) or not np.isfinite(v).all(): continue
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if s >= 1.80: continue
    (CAND if p.stem.startswith("p5_") else BASE)[p.stem] = v - v.mean()
print(f"в базисе {len(BASE)} векторов, кандидатов {len(CAND)}: {list(CAND)}")

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam*len(A)*np.eye(A.shape[1]); G[-1, -1] -= lam*len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
base_cols = list(BASE.values())
b = [fit_eval(base_cols, fm, em) for fm, em in ((half, ~half), (~half, half))]
print(f"\nстенд без кандидатов: {b[0]:.6f} / {b[1]:.6f}, среднее {np.mean(b):.6f}")

rs = np.random.default_rng(11)
print(f"\n{'кандидат':>16} {'A→B':>11} {'B→A':>11} {'среднее':>11} {'контроль':>11}")
for name, v in CAND.items():
    d, dc = [], []
    vp = rs.permutation(v)
    for i, (fm, em) in enumerate(((half, ~half), (~half, half))):
        d.append(fit_eval(base_cols + [v], fm, em) - b[i])
        dc.append(fit_eval(base_cols + [vp], fm, em) - b[i])
    print(f"{name:>16} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} {np.mean(dc):+11.2e}")

both = [v for k, v in CAND.items() if k.endswith("_q99")]
if len(both) == 2:
    d = [fit_eval(base_cols + both, fm, em) - b[i]
         for i, (fm, em) in enumerate(((half, ~half), (~half, half)))]
    print(f"{'обе q99 вместе':>16} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e}")
allc = list(CAND.values())
d = [fit_eval(base_cols + allc, fm, em) - b[i]
     for i, (fm, em) in enumerate(((half, ~half), (~half, half)))]
print(f"{'все четыре':>16} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e}")
print("\nотрицательное сверх контроля = кандидат несёт новое\ndone")
