"""Глубина окна или смысл события: корреляции решающей клетки.

v7d дал visit:256 = 1,74132 против visit:96 = 1,74850, то есть углубление окна улучшило
событийную модель на -0,00718 — больше, чем любая архитектурная правка (трансформер -0,00245,
лучший пуллинг -0,00197). Осталось понять, что стало с ортогональностью: рука была одна, и
таблицу корреляций скрипт не печатал.

Если декоррелированность даёт ГЛУБИНА, visit:256 уйдёт к 0,983 вслед за осью заказов — и тогда
у нас есть тот же вклад без её дефекта (15,5% клиентов без единого заказа).
Если останется около 0,988 — дело в СМЫСЛЕ события, и ось заказов незаменима.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from common import OUT, target, rmsle

y = target(348); ly = np.log1p(y)
N = [("visit:96", "v3_gru96_visit"), ("visit:256", "v7_visit_256"),
     ("ord:96", "v3_gru96_ord"), ("sord:96", "v5_gru96_sord")]
R = {}
for k, f in N:
    p = OUT / f"{f}.npy"
    if p.exists():
        R[k] = np.load(p).astype(np.float64)
    else:
        print(f"нет файла {f}.npy")
ref = np.load(OUT / "h8_gauss.npy").astype(np.float64)
base = R.get("visit:96")
ordv = R.get("ord:96")
print(f"{'рука':>10} {'RMSLE':>9} {'corr h8_gauss':>14} {'corr visit:96':>14} {'corr ord:96':>12}")
for k, v in R.items():
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    c1 = np.corrcoef(v, ref)[0, 1]
    c2 = np.corrcoef(v, base)[0, 1] if base is not None else float("nan")
    c3 = np.corrcoef(v, ordv)[0, 1] if ordv is not None else float("nan")
    print(f"{k:>10} {s:9.5f} {c1:14.5f} {c2:14.5f} {c3:12.5f}")
print("\nдиагностика: visit:96 обрезает окно у 43,6% клиентов, visit:256 у 4,3%, ord:96 у 1,1%")
print("done")
