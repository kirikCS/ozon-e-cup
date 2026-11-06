"""Четыре панели, которых не строят server_prep.py и server_prep2.py.

  P_scart   = search_to_cart   переходы «поиск -> корзина»
  P_sord    = search_to_ord    переходы «поиск -> заказ»
  P_sday    = индикатор дня с поиском
  P_catday  = индикатор дня с просмотром каталога

Все четыре выводятся прямо из сырых колонок train.parquet, без моделей.
Проверка: побитовое совпадение с рабочим кешем.
"""
import os
from datetime import date
from pathlib import Path
import numpy as np, polars as pl

SRC = os.environ.get("ECUP_SRC", "data/train.parquet")
CACHE = Path(os.environ.get("ECUP_CACHE", "cache"))
CACHE.mkdir(parents=True, exist_ok=True)
D0, N_DAYS = date(2025, 1, 1), 409

df = pl.read_parquet(SRC).select(
    ((pl.col("event_date") - pl.lit(D0)).dt.total_days()).cast(pl.Int32).alias("d"),
    pl.col("user_id").cast(pl.Int64),
    pl.col("search").cast(pl.Int64), pl.col("cat").cast(pl.Int64),
    pl.col("search_to_cart").cast(pl.Int64), pl.col("search_to_ord").cast(pl.Int64))

uids = np.load(CACHE / "uids.npy")
n_u = len(uids)
pos = {int(u): i for i, u in enumerate(uids)}
ui = np.fromiter((pos[int(u)] for u in df["user_id"].to_numpy()), np.int64, len(df))
flat = ui * N_DAYS + df["d"].to_numpy().astype(np.int64)

for name, col, dt, as_flag in (("scart", "search_to_cart", np.int16, False),
                               ("sord", "search_to_ord", np.int16, False),
                               ("sday", "search", np.int8, True),
                               ("catday", "cat", np.int8, True)):
    v = df[col].to_numpy().astype(np.float64)
    if as_flag: v = (v > 0).astype(np.float64)
    a = np.zeros(n_u * N_DAYS, np.float64)
    np.add.at(a, flat, v)
    if as_flag: a = np.minimum(a, 1)
    M = a.reshape(n_u, N_DAYS).astype(dt)
    np.save(CACHE / f"P_{name}.npy", M)
    print(f"saved {name}  ненулевых {100*(M>0).mean():.3f}%", flush=True)
print("done")
