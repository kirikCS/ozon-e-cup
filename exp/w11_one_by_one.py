"""Честный предельный вклад каждого кандидата ПООДИНОЧКЕ, без жадного отбора.

Нужен для одного конкретного решения: пересобирать ли компоненту v9_declm при батче 256.
При батче 1024 она даёт 1,74516, при 256 — 1,74338, то есть по скору лучше. Но за сутки шесть раз
подтвердилось, что лучший скор не означает большей пользы для стека, поэтому решать надо по вкладу.

Жадный отбор для этого не годится: он берёт по одному и порядок зависит от уже взятых. Здесь
каждый кандидат добавляется К БАЗИСУ ОТДЕЛЬНО, и отчёт идёт по половине B->A, на которой не
подбирались веса. Никакого выбора не делается вовсе, значит и проклятия победителя нет.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, NU, OUT

import re as _re
# ЛЮБОЙ вектор вида <буква><цифры>_... — кандидат. Перечисление префиксов подводило четырежды:
# забытый v11 не совпал с "v1_" и уехал В БАЗИС, изменив все вклады. Регулярка не забывается.
CANDRX = _re.compile(r"^[suvwxy]\d+_")

WHO = os.environ.get("WHO", "v9_declm_sel,v10_declm_96,v8_deep_sel,v2_gru96_declm,v7_visit_256")
LEAK = 1.70
MAXSC = float(os.environ.get("MAXSC", 1.80))   # потолок для БАЗИСА; кандидат меряется всегда
y = target(348); ly = np.log1p(y)
BASE = {}
PREF = tuple(f"{k}_" for k in ("s6","s7","s8","u1","u2","u3","u4","u5","u6",
                               "v1","v2","v3","v4","v5","v6","v7","v8","v9","u7","v10"))
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (NU,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= MAXSC or "fin" in p.stem or sc < LEAK or CANDRX.match(p.stem):
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

rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
REP = (~half, half)
cols = list(BASE.values())
b = fit_eval(cols, *REP)
rs = np.random.default_rng(11)
print(f"база (честная половина): {b:.6f}\n")
print(f"{'кандидат':>20} {'одиночный':>10} {'вклад':>11} {'контроль':>11} {'сверх':>11}")
for n in WHO.split(","):
    f = OUT / f"{n}.npy"
    if not f.exists():
        print(f"{n:>20}   нет файла")
        continue
    v = np.load(f).astype(np.float64)   # кандидат не фильтруется потолком
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    v = v - v.mean()
    d = fit_eval(cols + [v], *REP) - b
    dc = fit_eval(cols + [rs.permutation(v)], *REP) - b
    print(f"{n:>20} {sc:10.5f} {d:+11.2e} {dc:+11.2e} {d-dc:+11.2e}", flush=True)
print("\nсверх контроля — то, что кандидат несёт сам")
print("\ndone")
