"""Stage 3: core EDA analysis -> tables (CSV) + JSON summary."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("eda/cache")
OUT = Path("eda/out")
OUT.mkdir(exist_ok=True)
D0 = date(2025, 1, 1)
N_DAYS = 409
ANCHOR = 378          # 2026-01-14, validation anchor
H = 30

def dstr(i):
    return str(D0 + timedelta(days=int(i)))

GMV = np.load(CACHE / "P_gmv.npy")
GMVS = np.load(CACHE / "P_gmv_search.npy")
ORD = np.load(CACHE / "P_ord.npy")
SRCH = np.load(CACHE / "P_srch.npy")
CART = np.load(CACHE / "P_cart.npy")
VIS = np.load(CACHE / "P_visit.npy")
ENG = np.load(CACHE / "P_engaged.npy")
uids = np.load(CACHE / "uids.npy")
NU = GMV.shape[0]
res = {}

def rmsle(y, p):
    p = np.clip(p, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))

def wsum(M, a, b):
    """sum over days [a, b] inclusive"""
    return M[:, a:b + 1].sum(axis=1, dtype=np.float64)

# ============================================================ T1: overview
res["overview"] = {
    "n_rows": int(VIS.sum()),
    "n_users": NU,
    "n_days": N_DAYS,
    "date_min": dstr(0), "date_max": dstr(N_DAYS - 1),
    "panel_cells": NU * N_DAYS,
    "fill_rate_rows": float(VIS.mean()),
    "fill_rate_engaged": float(ENG.mean()),
    "rows_if_densified": NU * N_DAYS,
    "gmv_total": float(GMV.sum(dtype=np.float64)),
    "gmv_search_share": float(GMVS.sum(dtype=np.float64) / GMV.sum(dtype=np.float64)),
    "orders_total": int(ORD.sum(dtype=np.int64)),
    "searches_total": int(SRCH.sum(dtype=np.int64)),
    "carts_total": int(CART.sum(dtype=np.int64)),
    "buy_day_rate": float((GMV > 0).sum() / VIS.sum()),
    "test_window": ["2026-02-14", "2026-03-15"],
}

# ============================================================ T4: daily agg
daily = pl.DataFrame({
    "day_idx": np.arange(N_DAYS),
    "date": [dstr(i) for i in range(N_DAYS)],
    "visitors": VIS.sum(axis=0, dtype=np.int64),
    "engaged": ENG.sum(axis=0, dtype=np.int64),
    "buyers": (GMV > 0).sum(axis=0),
    "gmv": GMV.sum(axis=0, dtype=np.float64),
    "orders": ORD.sum(axis=0, dtype=np.int64),
    "carts": CART.sum(axis=0, dtype=np.int64),
    "searches": SRCH.sum(axis=0, dtype=np.int64),
}).with_columns(
    (pl.col("gmv") / pl.col("buyers")).alias("gmv_per_buyer"),
    (pl.col("gmv") / pl.col("visitors")).alias("gmv_per_visitor"),
    (pl.col("buyers") / pl.col("visitors")).alias("buyer_conv"),
    pl.col("date").str.to_date().dt.weekday().alias("dow"),
    pl.col("date").str.to_date().dt.strftime("%Y-%m").alias("month"),
)
daily.write_csv(OUT / "t04_daily_agg.csv")

# ============================================================ T5: monthly
monthly = daily.group_by("month").agg(
    pl.len().alias("days"),
    pl.col("gmv").sum(), pl.col("orders").sum(), pl.col("searches").sum(),
    pl.col("visitors").sum().alias("visit_days"),
    pl.col("buyers").sum().alias("buy_days"),
).sort("month").with_columns(
    (pl.col("gmv") / pl.col("days")).alias("gmv_per_day"),
    (pl.col("gmv") / pl.col("orders")).alias("aov"),
)
# unique active/buying users per month
mu, mb = [], []
for m in monthly["month"].to_list():
    y, mm = int(m[:4]), int(m[5:])
    s = (date(y, mm, 1) - D0).days
    e = (date(y + (mm == 12), (mm % 12) + 1, 1) - D0).days - 1
    e = min(e, N_DAYS - 1)
    mu.append(int((VIS[:, s:e + 1].sum(axis=1) > 0).sum()))
    mb.append(int((GMV[:, s:e + 1].sum(axis=1) > 0).sum()))
monthly = monthly.with_columns(pl.Series("uniq_active_users", mu), pl.Series("uniq_buyers", mb))
monthly.write_csv(OUT / "t05_monthly_agg.csv")

# ============================================================ T6: weekday
dow = daily.group_by("dow").agg(
    pl.col("gmv").mean().alias("gmv_mean"), pl.col("visitors").mean().alias("visitors_mean"),
    pl.col("buyers").mean().alias("buyers_mean"), pl.col("orders").mean().alias("orders_mean"),
    pl.col("searches").mean().alias("searches_mean"), pl.col("buyer_conv").mean(),
).sort("dow")
g0 = daily["gmv"].mean()
dow = dow.with_columns((pl.col("gmv_mean") / g0).alias("gmv_index"))
dow.write_csv(OUT / "t06_weekday_profile.csv")

# ============================================================ T7: user activity
u_vis = VIS.sum(axis=1)
u_eng = ENG.sum(axis=1)
u_buy = (GMV > 0).sum(axis=1)
u_gmv = GMV.sum(axis=1, dtype=np.float64)
u_ord = ORD.sum(axis=1, dtype=np.int64)
u_srch = SRCH.sum(axis=1, dtype=np.int64)
first = np.argmax(VIS > 0, axis=1)
last = N_DAYS - 1 - np.argmax(VIS[:, ::-1] > 0, axis=1)
has_any = u_vis > 0
recency = (N_DAYS - 1) - last

QS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
def qtab(name, arr):
    return {"metric": name, "mean": float(arr.mean()), "std": float(arr.std()),
            **{f"p{q}": float(np.percentile(arr, q)) for q in QS},
            "share_zero": float((arr == 0).mean())}
udist = pl.DataFrame([
    qtab("visit_days (rows per user)", u_vis.astype(float)),
    qtab("engaged_days", u_eng.astype(float)),
    qtab("buy_days", u_buy.astype(float)),
    qtab("lifetime_gmv", u_gmv),
    qtab("lifetime_orders", u_ord.astype(float)),
    qtab("lifetime_searches", u_srch.astype(float)),
    qtab("recency_days (from 2026-02-13)", recency.astype(float)),
    qtab("tenure_days (first->last)", (last - first).astype(float)),
])
udist.write_csv(OUT / "t07_user_distributions.csv")
res["users_never_active"] = int((~has_any).sum())
res["users_zero_lifetime_gmv"] = int((u_gmv == 0).sum())
res["users_zero_lifetime_gmv_share"] = float((u_gmv == 0).mean())

# ============================================================ T8: concentration
srt = np.sort(u_gmv)[::-1]
tot = srt.sum()
cum = np.cumsum(srt)
conc = []
for k in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    n = max(1, int(round(k * NU)))
    conc.append({"top_share_users": k, "n_users": n, "gmv_share": float(cum[n - 1] / tot)})
pl.DataFrame(conc).write_csv(OUT / "t08_gmv_concentration.csv")
asc = np.sort(u_gmv)
gini = float((2 * np.sum((np.arange(1, NU + 1)) * asc) / (NU * asc.sum())) - (NU + 1) / NU)
res["gini_lifetime_gmv"] = gini

# ============================================================ target / validation
A = ANCHOR
y = wsum(GMV, A + 1, A + H)                       # 2026-01-15 .. 2026-02-13
prev30 = wsum(GMV, A - 29, A)
prev60 = wsum(GMV, A - 59, A)
prev90 = wsum(GMV, A - 89, A)
prev180 = wsum(GMV, A - 179, A)
prev_all = wsum(GMV, 0, A)
ord30 = wsum(ORD, A - 29, A)
vis30 = wsum(VIS, A - 29, A)
vis90 = wsum(VIS, A - 89, A)
eng30 = wsum(ENG, A - 29, A)
srch30 = wsum(SRCH, A - 29, A)
cart30 = wsum(CART, A - 29, A)
buy30 = wsum((GMV > 0).astype(np.int8), A - 29, A)
buy_all = wsum((GMV > 0).astype(np.int8), 0, A)
lastA = np.where(VIS[:, :A + 1].max(axis=1) > 0, A - np.argmax(VIS[:, :A + 1][:, ::-1] > 0, axis=1), -1)
rec_A = np.where(lastA >= 0, A - lastA, 999)
lastbuyA = np.where((GMV[:, :A + 1] > 0).max(axis=1), A - np.argmax((GMV[:, :A + 1] > 0)[:, ::-1], axis=1), -1)
recbuy_A = np.where(lastbuyA >= 0, A - lastbuyA, 999)

res["target_at_anchor"] = {
    "anchor_date": dstr(A), "target_window": [dstr(A + 1), dstr(A + H)],
    "share_zero": float((y == 0).mean()),
    "mean": float(y.mean()), "std": float(y.std()),
    "sum": float(y.sum()),
    **{f"p{q}": float(np.percentile(y, q)) for q in [50, 75, 90, 95, 99, 99.9, 100]},
    "mean_log1p": float(np.log1p(y).mean()), "std_log1p": float(np.log1p(y).std()),
    "mean_pos": float(y[y > 0].mean()), "median_pos": float(np.median(y[y > 0])),
}

# ---- T9: target distribution buckets
edges = [0, 1e-9, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 1e18]
labels = ["0", "(0,10]", "(10,25]", "(25,50]", "(50,100]", "(100,250]", "(250,500]",
          "(500,1000]", "(1000,2500]", "(2500,5000]", ">5000"]
def bucketize(v):
    b = np.zeros(len(v), dtype=int)
    for i in range(1, len(edges) - 1):
        b[v > edges[i]] = i
    return b
by = bucketize(y)
tdist = pl.DataFrame({"bucket": [labels[i] for i in range(len(labels))],
                      "n_users": [int((by == i).sum()) for i in range(len(labels))]})
tdist = tdist.with_columns((pl.col("n_users") / NU).alias("share"),
                           pl.Series("gmv_sum", [float(y[by == i].sum()) for i in range(len(labels))]))
tdist = tdist.with_columns((pl.col("gmv_sum") / y.sum()).alias("gmv_share"))
tdist.write_csv(OUT / "t09_target_distribution.csv")

# ---- T10: target by recency bucket
rec_edges = [(0, 0), (1, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 180), (181, 400), (999, 999)]
rows = []
for lo, hi in rec_edges:
    m = (rec_A >= lo) & (rec_A <= hi)
    if m.sum() == 0: continue
    rows.append({"recency_bucket": f"{lo}-{hi}" if lo != 999 else "never active",
                 "n_users": int(m.sum()), "share_users": float(m.mean()),
                 "p_target_pos": float((y[m] > 0).mean()),
                 "mean_target": float(y[m].mean()),
                 "median_target_pos": float(np.median(y[m][y[m] > 0])) if (y[m] > 0).any() else 0.0,
                 "mean_log1p_target": float(np.log1p(y[m]).mean()),
                 "gmv_share": float(y[m].sum() / y.sum()),
                 "mean_prev30": float(prev30[m].mean())})
pl.DataFrame(rows).write_csv(OUT / "t10_target_by_recency.csv")

# ---- T11: target by prev30 GMV decile (zeros separate)
rows = []
mz = prev30 == 0
rows.append({"prev30_group": "prev30 = 0", "n_users": int(mz.sum()), "share_users": float(mz.mean()),
             "prev30_lo": 0.0, "prev30_hi": 0.0,
             "p_target_pos": float((y[mz] > 0).mean()), "mean_target": float(y[mz].mean()),
             "mean_log1p_target": float(np.log1p(y[mz]).mean()),
             "median_target_pos": float(np.median(y[mz][y[mz] > 0])) if (y[mz] > 0).any() else 0.0,
             "ratio_target_to_prev30": None})
pv = prev30[~mz]
qs = np.quantile(pv, np.linspace(0, 1, 11))
for i in range(10):
    lo, hi = qs[i], qs[i + 1]
    m = (~mz) & (prev30 >= lo) & (prev30 <= hi if i == 9 else prev30 < hi)
    if m.sum() == 0: continue
    rows.append({"prev30_group": f"decile {i+1}", "n_users": int(m.sum()), "share_users": float(m.mean()),
                 "prev30_lo": float(lo), "prev30_hi": float(hi),
                 "p_target_pos": float((y[m] > 0).mean()), "mean_target": float(y[m].mean()),
                 "mean_log1p_target": float(np.log1p(y[m]).mean()),
                 "median_target_pos": float(np.median(y[m][y[m] > 0])) if (y[m] > 0).any() else 0.0,
                 "ratio_target_to_prev30": float(y[m].sum() / prev30[m].sum())})
pl.DataFrame(rows).write_csv(OUT / "t11_target_by_prev30.csv")

# ---- T12: transition matrix prev30 bucket -> next30 bucket
bp = bucketize(prev30)
tm = np.zeros((len(labels), len(labels)))
for i in range(len(labels)):
    mi = bp == i
    if mi.sum() == 0: continue
    for j in range(len(labels)):
        tm[i, j] = (by[mi] == j).mean()
tmdf = pl.DataFrame({"prev30_bucket": labels, "n_users": [int((bp == i).sum()) for i in range(len(labels))]})
for j, lab in enumerate(labels):
    tmdf = tmdf.with_columns(pl.Series(f"->{lab}", tm[:, j]))
tmdf.write_csv(OUT / "t12_transition_matrix.csv")

# ---- correlations
lp, ly = np.log1p(prev30), np.log1p(y)
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
feats = {"prev30_gmv": prev30, "prev60_gmv": prev60, "prev90_gmv": prev90, "prev180_gmv": prev180,
         "lifetime_gmv": prev_all, "orders_30d": ord30, "buy_days_30d": buy30, "buy_days_life": buy_all,
         "visit_days_30d": vis30, "visit_days_90d": vis90, "engaged_days_30d": eng30,
         "searches_30d": srch30, "carts_30d": cart30,
         "recency_days": rec_A.astype(float), "recency_buy_days": np.minimum(recbuy_A, 400).astype(float)}
crow = []
for k, v in feats.items():
    crow.append({"feature": k,
                 "pearson_log1p_vs_log1p_target": float(np.corrcoef(np.log1p(np.abs(v)), ly)[0, 1]),
                 "spearman_vs_target": spearman(v, y),
                 "pearson_raw": float(np.corrcoef(v, y)[0, 1])})
pl.DataFrame(crow).sort("spearman_vs_target", descending=True).write_csv(OUT / "t18_feature_target_corr.csv")

# ---- T13: baselines RMSLE at anchor
base = {}
base["const 0"] = np.zeros(NU)
base["prev30 (naive AR = sample_submit)"] = prev30.copy()
base["prev60 / 2"] = prev60 / 2
base["prev90 / 3"] = prev90 / 3
base["prev180 / 6"] = prev180 / 6
base["lifetime mean per 30d"] = prev_all / ((A + 1) / 30)
base["mean(prev30, prev60/2, prev90/3)"] = (prev30 + prev60 / 2 + prev90 / 3) / 3
# optimal global constant
cs = np.linspace(0, 60, 601)
sc = [rmsle(y, np.full(NU, c)) for c in cs]
c_best = float(cs[int(np.argmin(sc))])
base[f"optimal constant ({c_best:.1f})"] = np.full(NU, c_best)
# optimal shrink alpha * prev30
als = np.linspace(0, 2, 201)
sa = [rmsle(y, a * prev30) for a in als]
a_best = float(als[int(np.argmin(sa))])
base[f"alpha*prev30 (alpha={a_best:.2f})"] = a_best * prev30
# optimal log-space linear: expm1(b*log1p(prev30) + c)
best = (1e9, None)
for b in np.linspace(0.2, 1.2, 51):
    for c in np.linspace(-0.5, 1.5, 41):
        p = np.expm1(b * lp + c)
        s = rmsle(y, p)
        if s < best[0]: best = (s, (b, c))
bb, cc = best[1]
base[f"expm1({bb:.2f}*log1p(prev30)+{cc:.2f})"] = np.expm1(bb * lp + cc)
# oracle: best constant per prev30 bucket (in log space -> mean of log1p)
pred_oracle = np.zeros(NU)
for i in range(len(labels)):
    m = bp == i
    if m.sum(): pred_oracle[m] = np.expm1(np.log1p(y[m]).mean())
base["oracle: mean log-target per prev30 bucket"] = pred_oracle
# oracle 2: bucket by (prev30 bucket x recency bucket)
recb = np.digitize(rec_A, [1, 4, 8, 15, 31, 61, 91, 181, 401])
pred_o2 = np.zeros(NU)
for i in range(len(labels)):
    for r in range(10):
        m = (bp == i) & (recb == r)
        if m.sum() >= 20: pred_o2[m] = np.expm1(np.log1p(y[m]).mean())
base["oracle: prev30 bucket x recency bucket"] = pred_o2

brows = [{"baseline": k, "rmsle": rmsle(y, v), "mean_pred": float(np.clip(v, 0, None).mean()),
          "share_pred_zero": float((np.clip(v, 0, None) == 0).mean())} for k, v in base.items()]
pl.DataFrame(brows).sort("rmsle").write_csv(OUT / "t13_baselines_rmsle.csv")
res["best_alpha_prev30"] = a_best
res["best_constant"] = c_best
res["best_log_linear"] = {"b": float(bb), "c": float(cc)}

# ============================================================ T14: seasonality YoY
def win(s, e):
    return float(GMV[:, s:e + 1].sum(dtype=np.float64))
segs = []
# test window analogue in 2025: 2026-02-14..2026-03-15 -> 2025-02-14..2025-03-15
pairs = [
    ("test window analogue 2025 (02-14..03-15)", (date(2025, 2, 14) - D0).days, (date(2025, 3, 15) - D0).days),
    ("preceding 30d 2025 (01-15..02-13)", (date(2025, 1, 15) - D0).days, (date(2025, 2, 13) - D0).days),
    ("validation target 2026 (01-15..02-13)", (date(2026, 1, 15) - D0).days, (date(2026, 2, 13) - D0).days),
    ("preceding 30d 2025-12-16..2026-01-14", (date(2025, 12, 16) - D0).days, (date(2026, 1, 14) - D0).days),
]
for nm, s, e in pairs:
    segs.append({"window": nm, "start": dstr(s), "end": dstr(e), "days": e - s + 1,
                 "gmv": win(s, e), "buyers": int((GMV[:, s:e + 1].sum(axis=1) > 0).sum()),
                 "active_users": int((VIS[:, s:e + 1].sum(axis=1) > 0).sum()),
                 "orders": int(ORD[:, s:e + 1].sum(dtype=np.int64))})
sdf = pl.DataFrame(segs)
sdf.write_csv(OUT / "t14_seasonality_windows.csv")
res["seasonal_ratio_2025_testwin_over_prev30"] = segs[0]["gmv"] / segs[1]["gmv"]

# rolling 30d GMV series (for chart)
cs30 = np.concatenate([[0], np.cumsum(GMV.sum(axis=0, dtype=np.float64))])
roll = np.array([cs30[i + 1] - cs30[max(0, i - 29)] for i in range(N_DAYS)])
pl.DataFrame({"date": [dstr(i) for i in range(N_DAYS)], "gmv_roll30": roll}).write_csv(OUT / "t14b_gmv_roll30.csv")

# ============================================================ T15: cohorts / retention
cohort_month = np.array([(D0 + timedelta(days=int(f))).strftime("%Y-%m") if h else "none"
                         for f, h in zip(first, has_any)])
months = sorted(set(cohort_month) - {"none"})
crows = []
for cm in months:
    m = cohort_month == cm
    crows.append({"cohort_first_seen": cm, "n_users": int(m.sum()),
                  "share": float(m.mean()),
                  "p_active_last30": float((vis30[m] > 0).mean()),
                  "p_buy_last30": float((prev30[m] > 0).mean()),
                  "mean_lifetime_gmv": float(u_gmv[m].mean()),
                  "mean_target": float(y[m].mean()),
                  "p_target_pos": float((y[m] > 0).mean())})
pl.DataFrame(crows).write_csv(OUT / "t15_cohorts.csv")

# retention: P(active in next 30d | active in prior 30d window), across anchors
ret = []
for a in range(59, N_DAYS - 30, 7):
    prev_act = VIS[:, a - 29:a + 1].sum(axis=1) > 0
    next_act = VIS[:, a + 1:a + 31].sum(axis=1) > 0
    prev_buy = GMV[:, a - 29:a + 1].sum(axis=1) > 0
    next_buy = GMV[:, a + 1:a + 31].sum(axis=1) > 0
    ret.append({"anchor": dstr(a),
                "n_active_prev30": int(prev_act.sum()),
                "p_active_next30_given_active": float(next_act[prev_act].mean()),
                "p_active_next30_given_inactive": float(next_act[~prev_act].mean()) if (~prev_act).any() else None,
                "n_buyers_prev30": int(prev_buy.sum()),
                "p_buy_next30_given_buy": float(next_buy[prev_buy].mean()),
                "p_buy_next30_given_nobuy": float(next_buy[~prev_buy].mean())})
pl.DataFrame(ret).write_csv(OUT / "t15b_retention.csv")

# ============================================================ T16: channel split
res["channel"] = {
    "gmv_search_share": float(GMVS.sum(dtype=np.float64) / GMV.sum(dtype=np.float64)),
    "gmv_cat_share": 1 - float(GMVS.sum(dtype=np.float64) / GMV.sum(dtype=np.float64)),
}

# ============================================================ T20: anchor stability
arow = []
for a in [378, 364, 350, 336, 322, 308, 294, 280, 250, 200, 150, 120]:
    yy = wsum(GMV, a + 1, a + H)
    pp = wsum(GMV, a - 29, a)
    arow.append({"anchor": dstr(a), "target_window": f"{dstr(a+1)}..{dstr(a+H)}",
                 "share_zero_target": float((yy == 0).mean()),
                 "mean_target": float(yy.mean()),
                 "sum_target": float(yy.sum()),
                 "rmsle_prev30": rmsle(yy, pp),
                 "rmsle_zero": rmsle(yy, np.zeros(NU)),
                 "best_alpha": float(np.linspace(0, 2, 201)[int(np.argmin([rmsle(yy, al * pp) for al in np.linspace(0, 2, 201)]))]),
                 "n_active_users_prev30": int((wsum(VIS, a - 29, a) > 0).sum())})
pl.DataFrame(arow).write_csv(OUT / "t20_anchor_stability.csv")

# ============================================================ T19: window activity
wrows = []
for w in [7, 14, 30, 60, 90, 180, 365, 409]:
    s = max(0, A + 1 - w)  # windows ending at anchor-day 408 (full history)
    s2 = N_DAYS - w
    act = (VIS[:, s2:].sum(axis=1) > 0)
    buy = (GMV[:, s2:].sum(axis=1) > 0)
    wrows.append({"window_days_to_2026-02-13": w, "start": dstr(s2),
                  "n_active": int(act.sum()), "share_active": float(act.mean()),
                  "n_buyers": int(buy.sum()), "share_buyers": float(buy.mean()),
                  "gmv": float(GMV[:, s2:].sum(dtype=np.float64))})
pl.DataFrame(wrows).write_csv(OUT / "t19_window_activity.csv")

# ============================================================ funnel
res["funnel"] = {
    "search_days": int((SRCH > 0).sum()),
    "searches_total": int(SRCH.sum(dtype=np.int64)),
    "cart_items": int(CART.sum(dtype=np.int64)),
    "ord_items": int(ORD.sum(dtype=np.int64)),
    "cart_per_search": float(CART.sum(dtype=np.int64) / SRCH.sum(dtype=np.int64)),
    "ord_per_cart": float(ORD.sum(dtype=np.int64) / CART.sum(dtype=np.int64)),
    "aov_item": float(GMV.sum(dtype=np.float64) / ORD.sum(dtype=np.int64)),
}

json.dump(res, open(OUT / "s03_summary.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str)[:6000])
print("\n--- baselines ---")
print(pl.read_csv(OUT / "t13_baselines_rmsle.csv"))
