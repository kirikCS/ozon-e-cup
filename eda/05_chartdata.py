"""Stage 5: histogram / curve data for the artifact + headroom estimate."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("eda/cache")
OUT = Path("eda/out")
D0 = date(2025, 1, 1); N = 409; A = 378
ds = lambda i: str(D0 + timedelta(days=int(i)))

GMV = np.load(CACHE / "P_gmv.npy"); VIS = np.load(CACHE / "P_visit.npy")
ORD = np.load(CACHE / "P_ord.npy"); SRCH = np.load(CACHE / "P_srch.npy"); CART = np.load(CACHE / "P_cart.npy")
NU = GMV.shape[0]
csG = np.cumsum(np.concatenate([np.zeros((NU, 1)), GMV.astype(np.float64)], 1), 1)
csV = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int32), VIS.astype(np.int32)], 1), 1)
csB = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int32), (GMV > 0).astype(np.int32)], 1), 1)
csO = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int64), ORD.astype(np.int64)], 1), 1)
wG = lambda a, b: csG[:, b + 1] - csG[:, a]
wV = lambda a, b: csV[:, b + 1] - csV[:, a]
wB = lambda a, b: csB[:, b + 1] - csB[:, a]
wO = lambda a, b: csO[:, b + 1] - csO[:, a]
def rmsle(y, p): return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(p, 0, None))) ** 2)))

ch = {}
# ---- daily series (downsample nothing, 409 pts is fine)
gmv_d = GMV.sum(0, dtype=np.float64); vis_d = VIS.sum(0, dtype=np.int64)
buy_d = (GMV > 0).sum(0); ord_d = ORD.sum(0, dtype=np.int64); srch_d = SRCH.sum(0, dtype=np.int64)
ch["daily"] = {"date": [ds(i) for i in range(N)], "gmv": gmv_d.round(1).tolist(),
               "visitors": vis_d.tolist(), "buyers": buy_d.tolist(),
               "orders": ord_d.tolist(), "searches": srch_d.tolist()}

# ---- rolling 30d
roll_g, roll_a, roll_b = [], [], []
for e in range(29, N):
    roll_g.append(float(wG(e - 29, e).sum())); roll_a.append(int((wV(e - 29, e) > 0).sum()))
    roll_b.append(int((wG(e - 29, e) > 0).sum()))
ch["roll30"] = {"date": [ds(i) for i in range(29, N)], "gmv": [round(x, 1) for x in roll_g],
                "active": roll_a, "buyers": roll_b}

# ---- target hist in log1p space
y = wG(A + 1, A + 30); p30 = wG(A - 29, A)
bins = np.linspace(0, 12, 49)
h, _ = np.histogram(np.log1p(y), bins=bins)
ch["target_log_hist"] = {"edges": bins.round(3).tolist(), "counts": h.tolist(),
                         "zero_count": int((y == 0).sum())}
hp, _ = np.histogram(np.log1p(p30), bins=bins)
ch["prev30_log_hist"] = {"edges": bins.round(3).tolist(), "counts": hp.tolist()}

# ---- lorenz
u_gmv = csG[:, -1]; s = np.sort(u_gmv)
cum = np.cumsum(s) / s.sum()
idx = np.linspace(0, NU - 1, 101).astype(int)
ch["lorenz"] = {"pop": (idx / (NU - 1)).round(4).tolist(), "gmv": cum[idx].round(5).tolist()}

# ---- activity-days histogram
u_vis = csV[:, -1]
h2, e2 = np.histogram(u_vis, bins=np.arange(0, 421, 10))
ch["visit_days_hist"] = {"edges": e2.tolist(), "counts": h2.tolist()}
u_buy = csB[:, -1]
h3, e3 = np.histogram(np.clip(u_buy, 0, 200), bins=np.arange(0, 205, 5))
ch["buy_days_hist"] = {"edges": e3.tolist(), "counts": h3.tolist()}

# ---- transition matrix (reuse labels)
edges = [0, 1e-9, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 1e18]
labels = ["0", "0–10", "10–25", "25–50", "50–100", "100–250", "250–500", "500–1k", "1k–2.5k", "2.5k–5k", ">5k"]
def bk(v):
    b = np.zeros(len(v), int)
    for i in range(1, len(edges) - 1): b[v > edges[i]] = i
    return b
bp, by = bk(p30), bk(y)
tm = np.zeros((len(labels), len(labels)))
for i in range(len(labels)):
    m = bp == i
    if m.sum():
        for j in range(len(labels)): tm[i, j] = (by[m] == j).mean()
ch["transition"] = {"labels": labels, "matrix": tm.round(4).tolist(),
                    "row_counts": [int((bp == i).sum()) for i in range(len(labels))]}

# ---- block activity
blk = []
for k in range(0, 13):
    e = N - 1 - 30 * k; st = e - 29
    if st < 0: break
    blk.append({"block": f"B-{k}", "start": ds(st), "end": ds(e),
                "share_active": round(float((wV(st, e) > 0).mean()), 4),
                "share_buyers": round(float((wG(st, e) > 0).mean()), 4)})
ch["blocks"] = blk[::-1]

# ---- interpurchase gap hist
buy = GMV > 0
gaps = []
for u in np.flatnonzero(buy.sum(1) >= 2):
    gaps.append(np.diff(np.flatnonzero(buy[u])))
gaps = np.concatenate(gaps)
h4, e4 = np.histogram(np.clip(gaps, 1, 90), bins=np.arange(1, 92, 2))
ch["gap_hist"] = {"edges": e4.tolist(), "counts": h4.tolist(),
                  "median": float(np.median(gaps)), "mean": float(gaps.mean())}

# ---- headroom: nested oracles
feats = {
    "prev30_bucket": bp,
    "recency": np.digitize(np.where(wV(A - 29, A) > 0, A - (A - np.argmax(VIS[:, :A + 1][:, ::-1] > 0, axis=1)), 999), [1, 4, 8, 15, 31]),
    "buydays90": np.digitize(wB(A - 89, A), [1, 2, 4, 7, 12, 20, 35]),
    "visitdays30": np.digitize(wV(A - 29, A), [1, 3, 6, 10, 15, 22, 28]),
    "ord30": np.digitize(wO(A - 29, A), [1, 2, 4, 7, 12, 25]),
}
def oracle(keys, min_n=25):
    code = np.zeros(NU, np.int64)
    for k in keys:
        code = code * 64 + feats[k]
    pred = np.zeros(NU)
    uc, inv = np.unique(code, return_inverse=True)
    ly = np.log1p(y)
    sums = np.bincount(inv, weights=ly); cnts = np.bincount(inv)
    glob = np.expm1(ly.mean())
    m = np.expm1(sums / np.maximum(cnts, 1))
    m[cnts < min_n] = glob
    pred = m[inv]
    return rmsle(y, pred), int(len(uc))
hr = []
for keys in [["prev30_bucket"], ["prev30_bucket", "recency"], ["prev30_bucket", "recency", "buydays90"],
             ["prev30_bucket", "recency", "buydays90", "visitdays30"],
             ["prev30_bucket", "recency", "buydays90", "visitdays30", "ord30"]]:
    r, k = oracle(keys)
    hr.append({"features": " x ".join(keys), "n_cells": k, "rmsle_in_sample_oracle": round(r, 4)})
pl.DataFrame(hr).write_csv(OUT / "t30_headroom_oracle.csv")
ch["headroom"] = hr

# ---- baselines for chart
bl = pl.read_csv(OUT / "t13_baselines_rmsle.csv").to_dicts()
ch["baselines"] = bl

# ---- feature corr for chart
ch["corr"] = pl.read_csv(OUT / "t18_feature_target_corr.csv").to_dicts()
ch["by_prev30"] = pl.read_csv(OUT / "t11_target_by_prev30.csv").to_dicts()
ch["by_recency"] = pl.read_csv(OUT / "t10_target_by_recency.csv").to_dicts()
ch["monthly"] = pl.read_csv(OUT / "t05_monthly_agg.csv").to_dicts()
ch["dow"] = pl.read_csv(OUT / "t06_weekday_profile.csv").to_dicts()
ch["seasonal"] = pl.read_csv(OUT / "t29_seasonal_calibration.csv").to_dicts()
ch["backtest"] = pl.read_csv(OUT / "t22_honest_backtest.csv").to_dicts()
ch["concentration"] = pl.read_csv(OUT / "t08_gmv_concentration.csv").to_dicts()
ch["target_dist"] = pl.read_csv(OUT / "t09_target_distribution.csv").to_dicts()
ch["cohorts"] = pl.read_csv(OUT / "t15_cohorts.csv").to_dicts()

json.dump(ch, open(OUT / "chartdata.json", "w"))
print("chartdata bytes:", (OUT / "chartdata.json").stat().st_size)
print(pl.DataFrame(hr))
EOF_MARK = None
