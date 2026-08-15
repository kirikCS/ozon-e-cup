"""Panels for the four raw columns that were never materialised: the catalog side of the funnel.

The original prep kept gmv, gmv_search, to_ord, to_cart, searches and two visit flags. That leaves
the channel decomposition of the funnel on the floor: to_cart = search_to_cart + cat_to_cart and
to_ord = search_to_ord + cat_to_ord hold exactly, and only the totals were ever built. The catalog
side is the small one — 4.0% of rows carry a catalog cart and 1.1% a catalog order — which is
precisely why no existing feature sees it.

Also kept separately: the `search` and `cat` visit flags, which the old prep OR-ed together into
`engaged` and thereby erased. 80.7% of active days are search days and 15.6% are catalog days, so
the mix between them is a real per-client dimension.

The `has_*` columns are not built: they are exactly (count > 0) on every one of the 30.6M rows.
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
    pl.col("cat_to_cart").cast(pl.Int32), pl.col("cat_to_ord").cast(pl.Int32),
    pl.col("gmv_cat").cast(pl.Float64),
    pl.col("search").cast(pl.Int8), pl.col("cat").cast(pl.Int8),
)
uids = np.load(CACHE / "uids.npy")
n_u = len(uids)
uidx = np.searchsorted(uids, df["user_id"].to_numpy()).astype(np.int64)
flat = uidx * N_DAYS + df["d"].to_numpy().astype(np.int64)
print(f"rows {df.height}, users {n_u}")

for name, col, dt in [("catcart", "cat_to_cart", np.int16), ("catord", "cat_to_ord", np.int16),
                      ("gmvcat", "gmv_cat", np.float32),
                      ("catflag", "cat", np.int8), ("srchflag", "search", np.int8)]:
    M = np.zeros(n_u * N_DAYS, dtype=dt)
    M[flat] = df[col].to_numpy().astype(dt)
    np.save(CACHE / f"P_{name}.npy", M.reshape(n_u, N_DAYS))
    print(f"saved {name} ({(M > 0).mean() * 100:.3f}% ненулевых ячеек)")
    del M
print("done")
