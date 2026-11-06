"""Зонд вдоль РАЗНОСТИ версий компоненты. Универсальный генератор.

Почему разность, а не сама компонента. Зонд трио объединил три новые компоненты как есть и дал
theta = 0,0035. Замена событийной компоненты, то есть направление РАЗНОСТИ старой и новой версии,
дала theta = 0,0055 — заметно больше. Объяснение простое: компонента почти целиком лежит в том,
что стек уже знает, и после ортогонализации от неё остаётся мало; разность же с самого начала
содержит только то, ЧТО ИЗМЕНИЛОСЬ.

Арифметика кампании. theta независимых направлений складывается в квадратуре, а выигрыш равен
theta^2 * kappa(2-kappa) / (2*R0). Чтобы закрыть разрыв 1,63e-05 до второго места, нужно
совокупное theta около 0,0104. Одна разность даёт 0,0055; четыре такой же силы дадут 0,011, и
заодно поднимут t до 1,5, где усадка Джеймса-Стайна перестаёт зануляять.

Поэтому имеет смысл мерить РАЗНЫЕ по природе изменения: длительность обучения, длину окна,
оператор смешивания, целевую функцию. Они меняют модель по-разному, значит их разности имеют
шанс быть слабо зависимыми.

Направление ортогонализуется к {1, q} опоры и нормируется на единичный разброс, поэтому нуль
арифметический: R'^2 = R0^2 + eps^2 при theta = 0.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
import polars as pl
from common import NU, OUT

EL = 2.3284
SRC = os.environ.get("SRC", "submissions/stack_v21.csv")
R0 = float(os.environ.get("R0", "1.6453455262403704"))
TARGET_COST = float(os.environ.get("COST", "0.0015"))
OLD = os.environ["OLD"]; NEW = os.environ["NEW"]; NAME = os.environ["NAME"]

a = np.load(OUT / f"{OLD}.npy").astype(np.float64)
b = np.load(OUT / f"{NEW}.npy").astype(np.float64)
d = (b - b.mean()) - (a - a.mean())
print(f"разность {NEW} − {OLD}: sd {d.std():.5f}, corr половин {np.corrcoef(a, b)[0,1]:.5f}")

df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert np.array_equal(uid, np.load("cache/uids.npy")), "порядок клиентов не совпадает"
A = np.stack([np.ones(NU), q], 1)
d = d - A @ np.linalg.solve(A.T @ A, A.T @ d)
sd0 = d.std(); d = d / sd0
print(f"после ортогонализации: доля вне прогноза {sd0/((b-b.mean())-(a-a.mean())).std():.3f}, "
      f"corr с q {np.corrcoef(d, q)[0,1]:+.1e}")

eps = float(np.sqrt((R0 + TARGET_COST) ** 2 - R0 ** 2))
t = q + eps * d
lo, hi = -6.0, 6.0
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
pred = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
out = f"submission/queue/probe_{NAME}.csv"
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\n{out}")
print(f"  eps {eps:.5f}, НУЛЬ {null:.7f}, уровень {np.log1p(pred).mean():.6f}")
print(f"  theta = (нуль^2 − замер^2) / (2*eps);  SE = {R0/np.sqrt(50000):.5f}")
print(f"  порог значимости |theta| > {R0/np.sqrt(50000):.4f} (это t = 1)")
print("\ndone")
