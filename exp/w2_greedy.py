"""Жадный отбор поверх стенда: что добавляет КАЖДЫЙ следующий вектор, а не каждый по отдельности.

w1 меряет кандидатов поодиночке, и этого мало для решения. v3_ord даёт −1,75e−03, v5_sord
−1,10e−03, но между собой они коррелируют 0,99415 — то есть могут нести одно направление, и
тогда второй не добавит ничего. Одиночные вклады этого не различают.

Здесь на каждом шаге к базису добавляется тот кандидат, который даёт наибольший ВЫИГРЫШ СВЕРХ
УЖЕ ВЗЯТЫХ, и печатается его предельный вклад. Шаг с вкладом около нуля означает, что дальше
брать нечего.

Порог остановки — не ноль, а цена лишнего столбца: она меряется тем же перемешанным контролем,
что и в w1. Кандидат принимается, только если обходит свой контроль.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

PREF = tuple(f"{k}_" for k in ("s6", "s7", "s8", "u1", "u2", "u3", "u4", "u5", "u6",
                               "v1", "v2", "v3", "v4", "v5", "v6", "v7"))
NSTEP = int(__import__("os").environ.get("NSTEP", 8))
LEAK_RMSLE = 1.70   # честный потолок: лучшая одиночная модель (лестница) даёт 1,73556

def leaky(name, score):
    """Вектор с ФИНАЛЬНОГО якоря 408 читает историю по 408-й день, а значит целевое окно
    якоря 348 (дни 349-378) лежит у него на ВХОДЕ. Такой вектор нельзя пускать ни в
    кандидаты, ни в базис. Ловим двумя способами сразу: по имени и по числу — имя можно
    забыть проставить, а скор 1,515 против честных 1,736 не подделаешь."""
    return ("fin" in name) or (score < LEAK_RMSLE)

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
        DROP.append(p.stem)
        continue
    (CAND if p.stem.startswith(PREF) else BASE)[p.stem] = v - v.mean()
print(f"ОТБРОШЕНО ПО УТЕЧКЕ {len(DROP)}: {DROP}")
print(f"в базисе {len(BASE)}, кандидатов {len(CAND)}", flush=True)

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
SPL = ((half, ~half), (~half, half))
cols = list(BASE.values())
cur = np.mean([fit_eval(cols, fm, em) for fm, em in SPL])
print(f"стенд без кандидатов: {cur:.6f}\n", flush=True)
rs = np.random.default_rng(11)
taken = []
for step in range(NSTEP):
    t0 = time.time()
    best = None
    for name, v in CAND.items():
        if name in taken:
            continue
        d = np.mean([fit_eval(cols + [v], fm, em) for fm, em in SPL]) - cur
        if best is None or d < best[0]:
            best = (d, name, v)
    d, name, v = best
    vp = rs.permutation(v)
    dc = np.mean([fit_eval(cols + [vp], fm, em) for fm, em in SPL]) - cur
    net = d - dc
    mark = "принят" if net < 0 else "ХУЖЕ СВОЕГО КОНТРОЛЯ — стоп"
    print(f"шаг {step+1}: {name:>22} предельный {d:+.2e}  контроль {dc:+.2e}  "
          f"сверх {net:+.2e}  [{mark}]  ({time.time()-t0:.0f}s)", flush=True)
    if net >= 0:
        break
    taken.append(name)
    cols.append(v)
    cur = cur + d
    print(f"          стенд стал {cur:.6f}, взято {len(taken)}", flush=True)
print(f"\nвзято: {taken}")
print(f"итог стенда: {cur:.6f}")
print("\ndone")
