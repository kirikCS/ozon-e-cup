"""Замена компоненты против добавления: разные операции с разной арифметикой.

u2_xdec — лучшая одиночная рука на календарной оси (1,73885 против 1,74198 у базы, -0,00313).
Но как НОВЫЙ СТОЛБЕЦ он даёт стенду лишь -1,14e-05: стек платит за новое направление, а не за
лучшую версию уже представленного семейства, и xdec к тому же ближе к стеку (0,99396 против
0,99297).

Замена — другая операция. Если вектор в базисе заменить на его улучшенную версию, выигрыш идёт
через ВЕС этой компоненты напрямую, а не через ортогональность. Ни один из прежних замеров этого
не проверял.

Отчёт по половине B->A. Кандидата на подмену выбираем по БЛИЗОСТИ к вектору, то есть по величине,
не зависящей от цели, — иначе это был бы отбор по тем же данным, по которым отчитываемся.

Обязательный контроль: замена делает ДВА действия сразу — добавляет новый вектор и убирает
старый. Убрать избыточный столбец само по себе может улучшить стенд. Поэтому рядом печатается
третья колонка: что будет, если просто УБРАТЬ вектор, ничего не добавив. Кандидату принадлежит
только разница между заменой и удалением.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

import re as _re
# ЛЮБОЙ вектор вида <буква><цифры>_... — кандидат. Перечисление префиксов подводило четырежды:
# забытый v11 не совпал с "v1_" и уехал В БАЗИС, изменив все вклады. Регулярка не забывается.
CANDRX = _re.compile(r"^[suvwxy]\d+_")

CAND_NAME = os.environ.get("CAND", "u2_xdec")
NTRY = int(os.environ.get("NTRY", 6))
LEAK_RMSLE = 1.70

def leaky(name, score):
    return ("fin" in name) or (score < LEAK_RMSLE)

PREF = tuple(f"{k}_" for k in ("s6", "s7", "s8", "u1", "u2", "u3", "u4", "u5", "u6",
                               "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "u7"))
y = target(348); ly = np.log1p(y)
BASE, CAND = {}, {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (250000,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= 1.80 or leaky(p.stem, sc):
        continue
    (CAND if CANDRX.match(p.stem) else BASE)[p.stem] = v - v.mean()
assert CAND_NAME in CAND, f"нет кандидата {CAND_NAME}"
cv = CAND[CAND_NAME]
print(f"кандидат {CAND_NAME}, в базисе {len(BASE)}", flush=True)

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
REP = (~half, half)
names = list(BASE)
cols = [BASE[n] for n in names]
b = fit_eval(cols, *REP)
add = fit_eval(cols + [cv], *REP)
print(f"базис как есть:   {b:.6f}")
print(f"базис + кандидат: {add:.6f}  ({add-b:+.2e})", flush=True)

cors = sorted(((abs(np.corrcoef(cv, BASE[n])[0, 1]), n) for n in names), reverse=True)
print(f"\n{'вектор':>22} {'corr':>8} {'замена':>11} {'просто убрать':>14} {'вклад кандидата':>16}")
for c, n in cors[:NTRY]:
    sw = [cv if m == n else BASE[m] for m in names]
    dr = [BASE[m] for m in names if m != n]
    r_sw = fit_eval(sw, *REP) - b
    r_dr = fit_eval(dr, *REP) - b
    print(f"{n:>22} {c:8.5f} {r_sw:+11.2e} {r_dr:+14.2e} {r_sw-r_dr:+16.2e}", flush=True)
print("\nпоследняя колонка = замена минус удаление: это и есть вклад САМОГО кандидата")
print(f"для сравнения, добавление дало {add-b:+.2e}")
print("\ndone")
