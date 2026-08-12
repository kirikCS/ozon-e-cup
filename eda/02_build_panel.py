"""Stage 2: build dense user x day panel matrices (250k x 409) and cache as .npy."""
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("/private/tmp/claude-501/-Users-kirillmakhankov-Documents-ozon-e-cup/808ae515-06c2-4407-a6fc-a50b3d5b6619/scratchpad/eda")
N_DAYS = 409

df = pl.read_parquet(CACHE / "compact.parquet")
print("loaded", df.shape)

uids = df["user_id"].unique().sort().to_numpy()
n_u = len(uids)
print("users", n_u)
np.save(CACHE / "uids.npy", uids)

# user_id -> row index
uidx = np.searchsorted(uids, df["user_id"].to_numpy()).astype(np.int32)
didx = df["d"].to_numpy().astype(np.int32)
flat = uidx.astype(np.int64) * N_DAYS + didx

specs = [
    ("gmv", "gmv", np.float32),
    ("gmv_search", "gmv_search", np.float32),
    ("ord", "to_ord", np.int16),
    ("srch", "searches", np.int16),
    ("cart", "to_cart", np.int16),
]
for name, col, dt in specs:
    M = np.zeros(n_u * N_DAYS, dtype=dt)
    M[flat] = df[col].to_numpy().astype(dt)
    M = M.reshape(n_u, N_DAYS)
    np.save(CACHE / f"P_{name}.npy", M)
    print("saved", name, M.dtype, M.nbytes / 1e6, "MB")
    del M

V = np.zeros(n_u * N_DAYS, dtype=np.int8)
V[flat] = 1
V = V.reshape(n_u, N_DAYS)
np.save(CACHE / "P_visit.npy", V)
print("saved visit", V.nbytes / 1e6, "MB", "fill rate", V.mean())

# also: "engaged" = row present AND (search or cat flag set)
E = np.zeros(n_u * N_DAYS, dtype=np.int8)
E[flat] = ((df["search"].to_numpy() | df["cat"].to_numpy()) > 0).astype(np.int8)
E = E.reshape(n_u, N_DAYS)
np.save(CACHE / "P_engaged.npy", E)
print("saved engaged, fill rate", E.mean())
print("done")
