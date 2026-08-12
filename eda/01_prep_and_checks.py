"""Stage 1: build a compact in-RAM cache + integrity / consistency checks."""
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

SRC = "data/train.parquet"
CACHE = Path("/private/tmp/claude-501/-Users-kirillmakhankov-Documents-ozon-e-cup/808ae515-06c2-4407-a6fc-a50b3d5b6619/scratchpad/eda")
OUT = Path("eda_out")
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

D0 = date(2025, 1, 1)
DMAX = date(2026, 2, 13)
N_DAYS = (DMAX - D0).days + 1

CNT = ["search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "searches"]
FLAG = ["search", "cat", "has_search_to_cart", "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord"]

res = {}

# ---------------------------------------------------------------- integrity
lf = pl.scan_parquet(SRC)

nulls = lf.select([pl.col(c).null_count().alias(c) for c in lf.collect_schema().names()]).collect(engine="streaming")
res["null_counts"] = nulls.to_dicts()[0]

dup = (
    lf.group_by(["user_id", "event_date"]).agg(pl.len().alias("n"))
    .filter(pl.col("n") > 1).select(pl.len()).collect(engine="streaming").item()
)
res["duplicate_user_date_pairs"] = int(dup)

# identities & flag consistency, single streaming pass
checks = lf.select(
    n_rows=pl.len(),
    gmv_identity_viol=((pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs() > 1e-6).sum(),
    cart_identity_viol=(pl.col("to_cart") != pl.col("search_to_cart") + pl.col("cat_to_cart")).sum(),
    ord_identity_viol=(pl.col("to_ord") != pl.col("search_to_ord") + pl.col("cat_to_ord")).sum(),
    flag_sc_viol=(pl.col("has_search_to_cart") != (pl.col("search_to_cart") > 0).cast(pl.Int64)).sum(),
    flag_so_viol=(pl.col("has_search_to_ord") != (pl.col("search_to_ord") > 0).cast(pl.Int64)).sum(),
    flag_cc_viol=(pl.col("has_cat_to_cart") != (pl.col("cat_to_cart") > 0).cast(pl.Int64)).sum(),
    flag_co_viol=(pl.col("has_cat_to_ord") != (pl.col("cat_to_ord") > 0).cast(pl.Int64)).sum(),
    no_channel_rows=((pl.col("search") == 0) & (pl.col("cat") == 0)).sum(),
    search_zero_but_activity=((pl.col("search") == 0) & ((pl.col("searches") > 0) | (pl.col("search_to_cart") > 0) | (pl.col("search_to_ord") > 0) | (pl.col("gmv_search") > 0))).sum(),
    cat_zero_but_activity=((pl.col("cat") == 0) & ((pl.col("cat_to_cart") > 0) | (pl.col("cat_to_ord") > 0) | (pl.col("gmv_cat") > 0))).sum(),
    search_flag_no_searches=((pl.col("search") == 1) & (pl.col("searches") == 0)).sum(),
    ord_but_zero_gmv=((pl.col("to_ord") > 0) & (pl.col("gmv") <= 0)).sum(),
    gmv_but_zero_ord=((pl.col("gmv") > 0) & (pl.col("to_ord") == 0)).sum(),
    ord_gt_cart=(pl.col("to_ord") > pl.col("to_cart")).sum(),
    s_ord_gt_s_cart=(pl.col("search_to_ord") > pl.col("search_to_cart")).sum(),
    c_ord_gt_c_cart=(pl.col("cat_to_ord") > pl.col("cat_to_cart")).sum(),
    both_channels=((pl.col("search") == 1) & (pl.col("cat") == 1)).sum(),
    only_search=((pl.col("search") == 1) & (pl.col("cat") == 0)).sum(),
    only_cat=((pl.col("search") == 0) & (pl.col("cat") == 1)).sum(),
    rows_gmv_pos=(pl.col("gmv") > 0).sum(),
    rows_ord_pos=(pl.col("to_ord") > 0).sum(),
    rows_cart_pos=(pl.col("to_cart") > 0).sum(),
    gmv_total=pl.col("gmv").sum(),
    gmv_search_total=pl.col("gmv_search").sum(),
    gmv_cat_total=pl.col("gmv_cat").sum(),
    ord_total=pl.col("to_ord").sum(),
    s_ord_total=pl.col("search_to_ord").sum(),
    c_ord_total=pl.col("cat_to_ord").sum(),
    cart_total=pl.col("to_cart").sum(),
    searches_total=pl.col("searches").sum(),
).collect(engine="streaming")
res["checks"] = checks.to_dicts()[0]

# ------------------------------------------------------------ column profile
prof_rows = []
num_cols = FLAG + CNT + ["gmv_search", "gmv_cat", "gmv"]
qs = [0.5, 0.9, 0.99, 0.999, 1.0]
agg = []
for c in num_cols:
    agg += [
        pl.col(c).mean().alias(f"{c}__mean"),
        pl.col(c).std().alias(f"{c}__std"),
        (pl.col(c) > 0).mean().alias(f"{c}__nonzero_share"),
        pl.col(c).sum().alias(f"{c}__sum"),
        pl.col(c).max().alias(f"{c}__max"),
    ] + [pl.col(c).quantile(q).alias(f"{c}__q{q}") for q in qs[:-1]]
pr = lf.select(agg).collect(engine="streaming").to_dicts()[0]
for c in num_cols:
    prof_rows.append({
        "column": c,
        "nonzero_share": pr[f"{c}__nonzero_share"],
        "mean": pr[f"{c}__mean"],
        "std": pr[f"{c}__std"],
        "p50": pr[f"{c}__q0.5"],
        "p90": pr[f"{c}__q0.9"],
        "p99": pr[f"{c}__q0.99"],
        "p99.9": pr[f"{c}__q0.999"],
        "max": pr[f"{c}__max"],
        "sum": pr[f"{c}__sum"],
    })
pl.DataFrame(prof_rows).write_csv(OUT / "t02_column_profile.csv")
res["column_profile"] = prof_rows

# ------------------------------------------------- conditional distributions
pos = lf.filter(pl.col("gmv") > 0)
gq = pos.select(
    [pl.col("gmv").quantile(q).alias(f"q{q}") for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]]
    + [pl.col("gmv").mean().alias("mean"), pl.col("gmv").std().alias("std"), pl.len().alias("n")]
).collect(engine="streaming")
res["gmv_positive_rows_dist"] = gq.to_dicts()[0]

# --------------------------------------------------------------- compact cache
compact = (
    lf.select(
        ((pl.col("event_date") - pl.lit(D0)).dt.total_days()).cast(pl.Int16).alias("d"),
        pl.col("user_id").cast(pl.Int32),
        *[pl.col(c).cast(pl.Int8) for c in FLAG],
        *[pl.col(c).cast(pl.Int16) for c in CNT],
        pl.col("gmv_search").cast(pl.Float32),
        pl.col("gmv_cat").cast(pl.Float32),
        pl.col("gmv").cast(pl.Float64),
    )
    .sort(["user_id", "d"])
)
compact.sink_parquet(CACHE / "compact.parquet", compression="zstd")
print("compact written")

res["meta"] = {"D0": str(D0), "DMAX": str(DMAX), "N_DAYS": N_DAYS}
json.dump(res, open(OUT / "s01_checks.json", "w"), indent=2, default=str)
print(json.dumps(res["checks"], indent=2, default=str))
print(json.dumps(res["gmv_positive_rows_dist"], indent=2, default=str))
