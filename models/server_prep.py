"""Rebuild the dense user x day panels from train.parquet (server-side, one shot)."""
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

SRC = os.environ.get("ECUP_SRC", "data/train.parquet")
CACHE = Path(os.environ.get("ECUP_CACHE", "cache"))
CACHE.mkdir(parents=True, exist_ok=True)
D0 = date(2025, 1, 1)
N_DAYS = 409

df = pl.read_parquet(SRC).select(
    ((pl.col("event_date") - pl.lit(D0)).dt.total_days()).cast(pl.Int32).alias("d"),
    pl.col("user_id").cast(pl.Int64),
    pl.col("search").cast(pl.Int8), pl.col("cat").cast(pl.Int8),
    pl.col("to_ord").cast(pl.Int32), pl.col("to_cart").cast(pl.Int32), pl.col("searches").cast(pl.Int32),
    pl.col("gmv").cast(pl.Float64), pl.col("gmv_search").cast(pl.Float64),
)
print("rows", df.height)
uids = df["user_id"].unique().sort().to_numpy()
n_u = len(uids)
np.save(CACHE / "uids.npy", uids)
uidx = np.searchsorted(uids, df["user_id"].to_numpy()).astype(np.int64)
flat = uidx * N_DAYS + df["d"].to_numpy().astype(np.int64)

for name, col, dt in [("gmv", "gmv", np.float32), ("gmv_search", "gmv_search", np.float32),
                      ("ord", "to_ord", np.int16), ("srch", "searches", np.int16), ("cart", "to_cart", np.int16)]:
    M = np.zeros(n_u * N_DAYS, dtype=dt)
    M[flat] = df[col].to_numpy().astype(dt)
    np.save(CACHE / f"P_{name}.npy", M.reshape(n_u, N_DAYS))
    print("saved", name)
    del M

V = np.zeros(n_u * N_DAYS, np.int8); V[flat] = 1
np.save(CACHE / "P_visit.npy", V.reshape(n_u, N_DAYS))
E = np.zeros(n_u * N_DAYS, np.int8)
E[flat] = ((df["search"].to_numpy() | df["cat"].to_numpy()) > 0).astype(np.int8)
np.save(CACHE / "P_engaged.npy", E.reshape(n_u, N_DAYS))
print("panels ready in", CACHE, "users", n_u)
