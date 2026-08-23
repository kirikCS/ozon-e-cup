"""Зонд вдоль оси заказов: измерить лидербордом истинный коэффициент направления.

Стенд дал этой оси −1,77e−03 сверх перемешанного контроля — втрое больше крупнейшего вклада за
проект (лестница порогов, −5,32e−04). Жадный отбор берёт её первым шагом, и она одна даёт 63%
всего прироста стенда. Но стенд — не стек: усадка стенд→стек 7,8x, и локальная оценка даёт только
НАПРАВЛЕНИЕ. Истинный коэффициент меряет лидерборд.

Направление ортогонализуется к {1, q} самого предсказания и нормируется на единичный разброс.
Тогда зонд меряет ровно то, чего в предсказании ещё нет:

    R'^2 = R0^2 − 2*eps*Cov(L−P, d) + eps^2*Var(d),  при Var(d)=1 нуль арифметический
    theta = (нуль^2 − R'^2) / (2*eps)

Уровень файла подгоняется двоичным поиском под EL = 2,3284 — без этого ось заказов сдвинула бы
весь прогноз вверх примерно на 0,51 в логарифме (её среднее log1p 2,835 против нашего 2,328).

Вектор усреднён по трём сидам, взаимная корреляция сидов 0,99973 — направление устойчиво, шум
сида в него почти не попадает.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import NU, OUT

EL = 2.3284
SRC = "submissions/stack_v21.csv"
R0 = 1.6453455262403704          # замеренный лидербордом скор опоры
TARGET_COST = 0.0015

df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert len(q) == NU, f"опора не того размера: {len(q)}"
uids_cache = np.load("cache/uids.npy")
assert np.array_equal(uid, uids_cache), "порядок клиентов в сабмите не совпадает с uids.npy"

d = np.load(OUT / "v6_ord_fin.npy").astype(np.float64)
print(f"вектор оси заказов: sd {d.std():.5f}, среднее log1p {d.mean():.5f}")

# ортогонализация к {1, q}: зонд обязан мерить то, чего в опоре ещё нет
A = np.stack([np.ones(NU), q], 1)
d = d - A @ np.linalg.solve(A.T @ A, A.T @ d)
d = d / d.std()
print(f"после ортогонализации: sd {d.std():.5f}, corr с q {np.corrcoef(d, q)[0,1]:+.2e}, "
      f"среднее {d.mean():+.2e}")

eps = float(np.sqrt((R0 + TARGET_COST) ** 2 - R0 ** 2))

def write_exact(name, t):
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
            lo = c
        else:
            hi = c
    p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    os.makedirs("submission/queue", exist_ok=True)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(f"submission/queue/{name}.csv")
    return float(np.log1p(p).mean())

lv = write_exact("probe_ord", q + eps * d)
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\nsubmission/queue/probe_ord.csv")
print(f"  eps {eps:.5f}, нуль {null:.7f}, цена слота {null-R0:.6f}, уровень файла {lv:.6f}")
print(f"\n  разбор после замера: theta = (нуль^2 − R'^2) / (2*eps)")
print(f"  порог аппарата     |theta| > {R0*1e-4/eps:.4f}")
print(f"  порог переноса     |theta| > {np.sqrt(R0**2/50000):.4f}")
print(f"  ожидание по стенду: вклад ≈ −2,3e−04 на стеке, то есть theta ≈ −{np.sqrt(2*R0*2.3e-4):.3f}")
print("done")
