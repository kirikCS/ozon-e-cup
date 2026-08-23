"""Стенд по всем ночным векторам сразу: один прогон вместо восьми.

Кандидаты — всё, что произвели ночные скрипты (s6, s7, u1-u4, v1, v2). Все они посчитаны на
якоре 348, том же, на котором стоит стенд, поэтому сравнимы напрямую — это проверено до запуска.

Аппарат тот же, что в p6: гребневая смесь всех прочих векторов на якоре 348 — стенд вместо стека,
веса на половине клиентов, скор на другой, обе стороны. Кандидат полезен, если добавление его
столбца улучшает скор на НЕВИДАННОЙ половине.

Контроль обязателен и не декоративен: тот же вектор, перемешанный по клиентам, несёт ровно один
лишний столбец и никакой информации. Он показывает цену самого расширения базиса, и сравнивать
надо с ним, а не с нулём.

Оговорка, которую нельзя забыть при чтении: стенд обязан повторять СЕМЕЙСТВЕННЫЙ состав
настоящего стека. Здесь кандидаты сплошь нейросетевые, и стенд тоже смещён в нейросети — на
лестнице порогов это уже привело к переоценке бустингового кандидата. Для нейросетевых кандидатов
смещение работает в другую сторону: их семейство в стенде представлено, значит оценка ближе к
честной, но усадка стенд->стек (7,8x) всё равно обязательна.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

# ВСЕ ночные префиксы. Пропущенный префикс не выпадает из замера — он попадает В СТЕНД,
# то есть кандидат оценивается против базиса, куда он сам уже включён. Список сверять при
# каждом новом скрипте.
PREF = tuple(f"{k}_" for k in
             ("s6", "s7", "s8", "u1", "u2", "u3", "u4", "u5", "u6",
              "v1", "v2", "v3", "v4", "v5"))
y = target(348); ly = np.log1p(y)
BASE, CAND, SCORE = {}, {}, {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (250000,) or not np.isfinite(v).all():
        continue
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if s >= 1.80:
        continue
    SCORE[p.stem] = s
    (CAND if p.stem.startswith(PREF) else BASE)[p.stem] = v - v.mean()
print(f"в базисе {len(BASE)} векторов, кандидатов {len(CAND)}")
for k in CAND:
    print(f"  {k:>22} одиночный RMSLE {SCORE[k]:.5f}")

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
base_cols = list(BASE.values())
SPL = ((half, ~half), (~half, half))
b = [fit_eval(base_cols, fm, em) for fm, em in SPL]
print(f"\nстенд без кандидатов: {b[0]:.6f} / {b[1]:.6f}, среднее {np.mean(b):.6f}", flush=True)

rs = np.random.default_rng(11)
print(f"\n{'кандидат':>22} {'A->B':>11} {'B->A':>11} {'среднее':>11} {'контроль':>11} {'сверх':>11}")
rows = []
for name, v in CAND.items():
    vp = rs.permutation(v)
    d = [fit_eval(base_cols + [v], fm, em) - b[i] for i, (fm, em) in enumerate(SPL)]
    dc = [fit_eval(base_cols + [vp], fm, em) - b[i] for i, (fm, em) in enumerate(SPL)]
    net = np.mean(d) - np.mean(dc)
    rows.append((net, name, d, np.mean(dc)))
    print(f"{name:>22} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} "
          f"{np.mean(dc):+11.2e} {net:+11.2e}", flush=True)

rows.sort()
good = [n for net, n, _, _ in rows if net < 0]
print(f"\nлучше своего контроля: {len(good)} из {len(CAND)}")
for net, n, _, _ in rows[:6]:
    print(f"  {n:>22} {net:+.2e}")

if len(good) > 1:
    cols = base_cols + [CAND[n] for n in good]
    d = [fit_eval(cols, fm, em) - b[i] for i, (fm, em) in enumerate(SPL)]
    print(f"\nвсе прошедшие вместе ({len(good)} шт.): {np.mean(d):+.2e}")
    print("  сравнивать с суммой одиночных: если совместный вклад заметно меньше,")
    print("  кандидаты несут ОДНО И ТО ЖЕ направление, а не разные")
    print(f"  сумма одиночных сверх контроля: {sum(net for net,_,_,_ in rows if net<0):+.2e}")
print("\nотрицательное сверх контроля = кандидат несёт новое\ndone")
