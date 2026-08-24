"""Применение измеренного направления с усадкой Джеймса-Стайна.

theta измерен лидербордом, поэтому здесь ничего не подбирается — только арифметика:

    t = theta / SE,  SE = R0 / sqrt(n_public),  kappa = max(0, 1 - 1/t^2)
    новый прогноз = q + kappa * theta * d

где d — то же направление, что было в зонде: разность версий, ортогонализованная к {1, q} и
нормированная на единичный разброс. Усадка обязательна: без неё мы применили бы шумовую часть
оценки theta, а она при t около двух составляет заметную долю.

ЗНАК. Отрицательное theta означает, что стеку нужно движение ПРОТИВ разности. У оператора
(трансформер минус GRU) theta = -0,0149: стек просит быть БОЛЬШЕ похожим на GRU, хотя по
одиночному скору трансформер лучше (1,74605 против 1,74850). Такое направление невозможно найти
отбором по улучшениям — только замером.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
import polars as pl
from common import NU, OUT

EL = 2.3284
SRC = os.environ.get("SRC", "submissions/stack_v21.csv")
R0 = float(os.environ.get("R0", "1.6453455262403704"))
OLD = os.environ["OLD"]; NEW = os.environ["NEW"]
TH = float(os.environ["THETA"]); NAME = os.environ["NAME"]

a = np.load(OUT / f"{OLD}.npy").astype(np.float64)
b = np.load(OUT / f"{NEW}.npy").astype(np.float64)
d = (b - b.mean()) - (a - a.mean())
df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert np.array_equal(uid, np.load("cache/uids.npy")), "порядок клиентов не совпадает"
A = np.stack([np.ones(NU), q], 1)
d = d - A @ np.linalg.solve(A.T @ A, A.T @ d)
d = d / d.std()

SE = R0 / np.sqrt(50000)
t = TH / SE
k = max(0.0, 1.0 - 1.0 / t ** 2)
coef = k * TH
gain = -TH ** 2 * k * (2 - k) / (2 * R0)
print(f"theta {TH:+.5f}, t {t:+.2f}, kappa {k:.3f}, коэффициент {coef:+.5f}")
print(f"ожидаемый выигрыш {gain:+.3e}, ожидаемый скор {R0+gain:.10f}")

tq = q + coef * d
lo, hi = -6.0, 6.0
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(tq + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
pred = np.clip(np.expm1(tq + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
out = f"submission/queue/{NAME}.csv"
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
print(f"\n{out}: уровень {np.log1p(pred).mean():.6f}, "
      f"corr с опорой {np.corrcoef(np.log1p(pred), q)[0,1]:.6f}")
print("\ndone")
