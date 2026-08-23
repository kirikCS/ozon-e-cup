"""Направление из двух подтверждённых компонент под зонд.

Честная проверка на отложенной половине (w7, w9) оставила из ~40 рук два вектора, несущих почти
весь выигрыш набора: v9_declm (причинный трансформер над событиями, -1,62e-04) и v8_deep
(окно 256 вместо 96, -1,14e-04). Оба посчитаны на обоих протоколах и усреднены по трём сидам.

Компоненты входят в стек не переподгонкой весов, а ЗОНДОМ: строится направление, лидерборд меряет
его истинный коэффициент theta, и оно применяется с усадкой Джеймса-Стайна. Так собраны v18-v21.

Веса комбинации подбираются на якоре 348 по половине B, которая НЕ участвовала в отборе кандидатов
(отбор шёл по A). Иначе веса подгонялись бы под те же данные, что и сам выбор этих двух векторов.

Теми же весами сворачиваются ФИНАЛЬНЫЕ векторы. Результат ортогонализуется к {1, q} опоры —
зонд обязан мерить то, чего в stack_v21 ещё нет, а не то, что там уже есть.

Ожидание, записанное ДО замера: набор даёт -3,38e-04 на честной половине стенда, после усадки
стенд->стек 7,8x это около -4,3e-05, чему соответствует theta ~ 0,0125. Пороги аппарата:
измеримость 0,0023, перенос на приват 0,0074.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
import polars as pl
from pathlib import Path
from common import target, rmsle, NU, OUT

import re as _re
# ЛЮБОЙ вектор вида <буква><цифры>_... — кандидат. Перечисление префиксов подводило четырежды:
# забытый v11 не совпал с "v1_" и уехал В БАЗИС, изменив все вклады. Регулярка не забывается.
CANDRX = _re.compile(r"^[suvwx]\d+_")

EL = 2.3284
SRC = "submissions/stack_v21.csv"
R0 = 1.6453455262403704
TARGET_COST = 0.0015
PAIR = os.environ.get("PAIR", "v9_declm,v8_deep").split(",")
LEAK_RMSLE = 1.70

y = target(348); ly = np.log1p(y)
BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (NU,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= 1.80 or "fin" in p.stem or sc < LEAK_RMSLE:
        continue
    if CANDRX.match(p.stem):
        continue
    BASE[p.stem] = v - v.mean()
print(f"в базисе {len(BASE)}", flush=True)

SEL = [np.load(OUT / f"{n}_sel.npy").astype(np.float64) for n in PAIR]
FIN = [np.load(OUT / f"{n}_fin.npy").astype(np.float64) for n in PAIR]
for n, v in zip(PAIR, SEL):
    sh = v - v.mean() + ly.mean()
    print(f"  {n}: SELECT RMSLE {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}")

rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
fm = ~half            # веса подбираем на B — половине, не участвовавшей в отборе кандидатов
cols = list(BASE.values()) + [v - v.mean() for v in SEL]
M = np.stack(cols, 1)
A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
lam = 1e-4
G = A.T @ A + lam * len(A) * np.eye(A.shape[1]); G[-1, -1] -= lam * len(A)
w = np.linalg.solve(G, A.T @ ly[fm])
wpair = w[len(BASE):len(BASE) + len(PAIR)]
print(f"веса пары в ридже: {dict(zip(PAIR, np.round(wpair, 5)))}", flush=True)

d = sum(wi * (v - v.mean()) for wi, v in zip(wpair, FIN))
df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert np.array_equal(uid, np.load("cache/uids.npy")), "порядок клиентов не совпадает"
Aq = np.stack([np.ones(NU), q], 1)
d = d - Aq @ np.linalg.solve(Aq.T @ Aq, Aq.T @ d)
d = d / d.std()
print(f"направление: sd {d.std():.5f}, corr с q {np.corrcoef(d, q)[0,1]:+.2e}")

eps = float(np.sqrt((R0 + TARGET_COST) ** 2 - R0 ** 2))
lo, hi = -6.0, 6.0
t = q + eps * d
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
pred = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv("submission/queue/probe_deep_declm.csv")
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\nsubmission/queue/probe_deep_declm.csv")
print(f"  eps {eps:.5f}, нуль {null:.7f}, уровень файла {np.log1p(pred).mean():.6f}")
print(f"  порог аппарата |theta| > {R0*1e-4/eps:.4f}")
print(f"  порог переноса |theta| > {np.sqrt(R0**2/50000):.4f}")
print(f"  ожидание по стенду theta ~ 0,0125, то есть скор около {np.sqrt(null**2 - 2*eps*0.0125):.6f}")
print("\ndone")
