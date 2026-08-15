"""Build the channel-funnel panels that server_prep.py never extracted.

train.parquet carries four columns absent from cache/: search_to_cart, search_to_ord and
the per-channel day flags search / cat. Their catalog counterparts derive exactly
(cat_to_x = to_x - search_to_x), so only the search side plus the two flags are stored:

  P_scart.npy    int16  search_to_cart per day
  P_sord.npy     int16  search_to_ord per day
  P_sday.npy     int8   1 if the user searched that day
  P_catday.npy   int8   1 if the user browsed the catalog that day

Row order matches uids.npy from server_prep.py exactly (verified against it, not rebuilt).
"""
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

SRC = os.environ.get("ECUP_SRC", "data/train.parquet")
CACHE = Path(os.environ.get("ECUP_CACHE", "cache"))
D0 = date(2025, 1, 1)
N_DAYS = 409

df = pl.read_parquet(SRC).select(
    ((pl.col("event_date") - pl.lit(D0)).dt.total_days()).cast(pl.Int32).alias("d"),
    pl.col("user_id").cast(pl.Int64),
    pl.col("search_to_cart").cast(pl.Int32), pl.col("search_to_ord").cast(pl.Int32),
    pl.col("search").cast(pl.Int8), pl.col("cat").cast(pl.Int8),
    pl.col("to_cart").cast(pl.Int32), pl.col("to_ord").cast(pl.Int32),
)
print("rows", df.height)

uids = np.load(CACHE / "uids.npy")
uidx = np.searchsorted(uids, df["user_id"].to_numpy()).astype(np.int64)
assert (uids[uidx] == df["user_id"].to_numpy()).all(), "user_id outside uids.npy"
flat = uidx * N_DAYS + df["d"].to_numpy().astype(np.int64)

# identities from the EDA must hold exactly, otherwise the raw file is not what we think
assert (df["search_to_cart"].to_numpy() <= df["to_cart"].to_numpy()).all()
assert (df["search_to_ord"].to_numpy() <= df["to_ord"].to_numpy()).all()

for name, col, dt in [("scart", "search_to_cart", np.int16), ("sord", "search_to_ord", np.int16),
                      ("sday", "search", np.int8), ("catday", "cat", np.int8)]:
    v = df[col].to_numpy()
    assert v.max() <= np.iinfo(dt).max, f"{col} overflows {dt}"
    M = np.zeros(len(uids) * N_DAYS, dtype=dt)
    M[flat] = v.astype(dt)
    np.save(CACHE / f"P_{name}.npy", M.reshape(len(uids), N_DAYS))
    print("saved", name, "max", int(v.max()))
    del M

print("fx panels ready in", CACHE)
