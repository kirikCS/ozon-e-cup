"""Stage 4: selection-bias diagnosis, honest analogue backtest, extra distributions."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("/private/tmp/claude-501/-Users-kirillmakhankov-Documents-ozon-e-cup/808ae515-06c2-4407-a6fc-a50b3d5b6619/scratchpad/eda")
OUT = Path("eda_out")
D0 = date(2025, 1, 1)
N = 409
ds = lambda i: str(D0 + timedelta(days=int(i)))

GMV = np.load(CACHE / "P_gmv.npy")
GMVS = np.load(CACHE / "P_gmv_search.npy")
VIS = np.load(CACHE / "P_visit.npy")
ORD = np.load(CACHE / "P_ord.npy")
SRCH = np.load(CACHE / "P_srch.npy")
CART = np.load(CACHE / "P_cart.npy")
NU = GMV.shape[0]
res = {}

def rmsle(y, p, m=None):
    if m is not None: y, p = y[m], p[m]
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(p, 0, None))) ** 2)))

csV = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int32), VIS.astype(np.int32)], 1), 1)
csG = np.cumsum(np.concatenate([np.zeros((NU, 1), np.float64), GMV.astype(np.float64)], 1), 1)
csO = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int64), ORD.astype(np.int64)], 1), 1)
csB = np.cumsum(np.concatenate([np.zeros((NU, 1), np.int32), (GMV > 0).astype(np.int32)], 1), 1)
wV = lambda a, b: csV[:, b + 1] - csV[:, a]
wG = lambda a, b: csG[:, b + 1] - csG[:, a]
wO = lambda a, b: csO[:, b + 1] - csO[:, a]
wB = lambda a, b: csB[:, b + 1] - csB[:, a]

# ================================================== 1. selection rule evidence
sel = []
for k in range(0, 13):
    e = N - 1 - 30 * k; s = e - 29
    if s < 0: break
    sel.append({"block": f"-{k}", "start": ds(s), "end": ds(e),
                "n_active": int((wV(s, e) > 0).sum()), "share_active": float((wV(s, e) > 0).mean()),
                "n_buyers": int((wG(s, e) > 0).sum()), "share_buyers": float((wG(s, e) > 0).mean()),
                "gmv": float(wG(s, e).sum())})
pl.DataFrame(sel).write_csv(OUT / "t21_block_activity.csv")

# ================================================== 2. honest analogue backtest
def backtest(E, apply_rule=True, name=""):
    """cohort = active in each of 3 x 30d blocks ending at E; target = [E+1, E+30]"""
    if apply_rule:
        m = (wV(E - 29, E) > 0) & (wV(E - 59, E - 30) > 0) & (wV(E - 89, E - 60) > 0)
    else:
        m = np.ones(NU, bool)
    y = wG(E + 1, E + 30)
    p30 = wG(E - 29, E)
    p60 = wG(E - 59, E)
    p90 = wG(E - 89, E)
    act_next = wV(E + 1, E + 30) > 0
    lp = np.log1p(p30)
    # fit log-linear on the cohort itself (in-sample optimum = upper bound for that form)
    best = (1e9, None)
    for b in np.linspace(0.2, 1.2, 21):
        for c in np.linspace(-0.5, 1.5, 21):
            s = rmsle(y, np.expm1(b * lp + c), m)
            if s < best[0]: best = (s, (b, c))
    als = np.linspace(0, 2, 101)
    sa = [rmsle(y, a * p30, m) for a in als]
    return {
        "setup": name, "anchor": ds(E), "target_window": f"{ds(E+1)}..{ds(E+30)}",
        "cohort_size": int(m.sum()), "cohort_share_of_250k": float(m.mean()),
        "p_active_in_target": float(act_next[m].mean()),
        "share_zero_target": float((y[m] == 0).mean()),
        "mean_target": float(y[m].mean()), "median_target_pos": float(np.median(y[m][y[m] > 0])),
        "mean_log1p_target": float(np.log1p(y[m]).mean()),
        "rmsle_zero": rmsle(y, np.zeros(NU), m),
        "rmsle_prev30": rmsle(y, p30, m),
        "rmsle_prev90_3": rmsle(y, p90 / 3, m),
        "best_alpha": float(als[int(np.argmin(sa))]), "rmsle_best_alpha": float(min(sa)),
        "loglin_b": best[1][0], "loglin_c": best[1][1], "rmsle_loglin": best[0],
    }

bt = [
    backtest(378, True, "LB-like anchor (target = selection window!)"),
    backtest(348, True, "target inside guaranteed-active zone"),
    backtest(318, True, "target inside guaranteed-active zone"),
    backtest(288, True, "HONEST: rule replicated, target free"),
    backtest(258, True, "HONEST: rule replicated, target free"),
    backtest(228, True, "HONEST: rule replicated, target free"),
    backtest(288, False, "no rule (all 250k), target free"),
    backtest(258, False, "no rule (all 250k), target free"),
]
pl.DataFrame(bt).write_csv(OUT / "t22_honest_backtest.csv")
res["backtest"] = bt

# ================================================== 3. block-to-block GMV autocorr
blocks = []
for k in range(12, -1, -1):
    e = N - 1 - 30 * k; s = e - 29
    if s < 0: continue
    blocks.append((f"B-{k}", wG(s, e)))
names = [b[0] for b in blocks]
M = np.array([np.log1p(b[1]) for b in blocks])
cm = np.corrcoef(M)
acdf = pl.DataFrame({"block": names})
for j, n in enumerate(names):
    acdf = acdf.with_columns(pl.Series(n, cm[:, j]))
acdf.write_csv(OUT / "t23_block_autocorr_log1p.csv")
# lag profile vs last block
last = np.log1p(blocks[-1][1])
lag = [{"lag_blocks_30d": len(blocks) - 1 - i, "block": names[i],
        "pearson_log1p_with_last_block": float(np.corrcoef(np.log1p(blocks[i][1]), last)[0, 1])}
       for i in range(len(blocks) - 1)]
pl.DataFrame(lag).sort("lag_blocks_30d").write_csv(OUT / "t23b_lag_profile.csv")

# ================================================== 4. seasonality, detrended
# GMV per active user in a 30d window, rolling -> removes user-base growth partially
rows = []
for e in range(29, N):
    s = e - 29
    a = (wV(s, e) > 0).sum()
    rows.append({"end_date": ds(e), "gmv_30d": float(wG(s, e).sum()), "active_30d": int(a),
                 "gmv_per_active_30d": float(wG(s, e).sum() / a),
                 "buyers_30d": int((wG(s, e) > 0).sum())})
season = pl.DataFrame(rows)
season.write_csv(OUT / "t24_rolling30_windows.csv")
# YoY: same calendar window 2025 vs the 2026 equivalent where available
def cwin(y1, m1, d1, y2, m2, d2):
    s = (date(y1, m1, d1) - D0).days; e = (date(y2, m2, d2) - D0).days
    a = (wV(s, e) > 0).sum()
    return {"window": f"{ds(s)}..{ds(e)}", "gmv": float(wG(s, e).sum()), "active": int(a),
            "gmv_per_active": float(wG(s, e).sum() / a), "buyers": int((wG(s, e) > 0).sum()),
            "aov": float(wG(s, e).sum() / max(1, wO(s, e).sum()))}
yoy = [
    {"label": "TEST-window analogue 2025 (Feb14-Mar15)", **cwin(2025, 2, 14, 2025, 3, 15)},
    {"label": "prev 30d 2025 (Jan15-Feb13)", **cwin(2025, 1, 15, 2025, 2, 13)},
    {"label": "next 30d 2025 (Mar16-Apr14)", **cwin(2025, 3, 16, 2025, 4, 14)},
    {"label": "VALID target 2026 (Jan15-Feb13)", **cwin(2026, 1, 15, 2026, 2, 13)},
    {"label": "prev 30d 2026 (Dec16-Jan14)", **cwin(2025, 12, 16, 2026, 1, 14)},
]
pl.DataFrame(yoy).write_csv(OUT / "t25_yoy_seasonality.csv")
res["season_2025_test_vs_prev"] = yoy[0]["gmv"] / yoy[1]["gmv"]
res["season_2025_test_vs_prev_per_active"] = yoy[0]["gmv_per_active"] / yoy[1]["gmv_per_active"]

# ================================================== 5. interpurchase time
buy = GMV > 0
rows = []
nb = buy.sum(1)
sel_u = np.where(nb >= 2)[0]
gaps_all = []
for chunk in np.array_split(sel_u, 20):
    for u in chunk:
        idx = np.flatnonzero(buy[u])
        gaps_all.append(np.diff(idx))
gaps = np.concatenate(gaps_all)
res["interpurchase_gap_days"] = {"n": int(len(gaps)), "mean": float(gaps.mean()),
                                 **{f"p{q}": float(np.percentile(gaps, q)) for q in [10, 25, 50, 75, 90, 95, 99]}}
# per-user mean gap
mg = np.array([g.mean() if len(g) else np.nan for g in gaps_all])
res["user_mean_interpurchase_gap"] = {f"p{q}": float(np.nanpercentile(mg, q)) for q in [10, 25, 50, 75, 90, 95, 99]}

# ================================================== 6. per-user channel + AOV
u_gmv = csG[:, -1]; u_gmvs = GMVS.sum(1, dtype=np.float64); u_ord = csO[:, -1]
share_s = np.where(u_gmv > 0, u_gmvs / np.maximum(u_gmv, 1e-9), np.nan)
aov = np.where(u_ord > 0, u_gmv / np.maximum(u_ord, 1), np.nan)
ch = []
for lo, hi, lab in [(-.01, .01, "0% search (catalog-only buyer)"), (.01, .5, "1-50% search"),
                    (.5, .9, "50-90% search"), (.9, .999, "90-99.9% search"), (.999, 1.01, "100% search")]:
    m = (share_s > lo) & (share_s <= hi)
    ch.append({"search_gmv_share_bucket": lab, "n_users": int(m.sum()),
               "share_of_buyers": float(m.sum() / (u_gmv > 0).sum()),
               "mean_lifetime_gmv": float(u_gmv[m].mean()) if m.any() else 0.0})
pl.DataFrame(ch).write_csv(OUT / "t26_channel_mix.csv")
res["aov_per_user"] = {f"p{q}": float(np.nanpercentile(aov, q)) for q in [5, 25, 50, 75, 95, 99]}
res["gmv_per_order_overall"] = float(u_gmv.sum() / u_ord.sum())

# ================================================== 7. funnel by window
fr = []
for lab, s, e in [("last 30d", N - 30, N - 1), ("last 90d", N - 90, N - 1), ("full", 0, N - 1)]:
    sr = SRCH[:, s:e + 1].sum(dtype=np.int64); ca = CART[:, s:e + 1].sum(dtype=np.int64)
    orr = ORD[:, s:e + 1].sum(dtype=np.int64); gv = GMV[:, s:e + 1].sum(dtype=np.float64)
    fr.append({"window": lab, "searches": int(sr), "cart_items": int(ca), "ord_items": int(orr),
               "cart_per_search": ca / sr, "ord_per_cart": orr / ca, "gmv": gv, "gmv_per_item": gv / orr})
pl.DataFrame(fr).write_csv(OUT / "t27_funnel.csv")

# ================================================== 8. zero-target profile
A = 378
y = wG(A + 1, A + 30)
p30 = wG(A - 29, A)
v30 = wV(A - 29, A)
b90 = wB(A - 89, A)
zp = []
for lab, m in [("target = 0", y == 0), ("target > 0", y > 0)]:
    zp.append({"group": lab, "n_users": int(m.sum()), "share": float(m.mean()),
               "mean_prev30_gmv": float(p30[m].mean()), "share_prev30_gmv_zero": float((p30[m] == 0).mean()),
               "mean_visit_days_30d": float(v30[m].mean()),
               "mean_buy_days_90d": float(b90[m].mean()),
               "mean_lifetime_gmv": float(csG[m, A + 1].mean())})
pl.DataFrame(zp).write_csv(OUT / "t28_zero_vs_pos_profile.csv")

json.dump(res, open(OUT / "s04_summary.json", "w"), indent=2, default=str)
pl.Config.set_tbl_width_chars(240); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(25)
print(pl.DataFrame(sel)); print()
print(pl.DataFrame(bt).select("setup", "anchor", "cohort_size", "p_active_in_target", "share_zero_target",
                              "mean_target", "rmsle_zero", "rmsle_prev30", "best_alpha", "rmsle_best_alpha", "rmsle_loglin"))
print(json.dumps({k: v for k, v in res.items() if k != "backtest"}, indent=2, default=str))
