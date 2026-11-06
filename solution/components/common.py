"""Shared infrastructure: dense panel access, feature store, validation protocol, scoreboard."""
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import os
# Пути задаёт run.py через окружение. Значения по умолчанию — относительно
# текущей директории, чтобы скрипт можно было запустить и в одиночку.
CACHE = Path(os.environ.get("ECUP_CACHE", "cache"))    # панели, .npy
FS = Path(os.environ.get("ECUP_FS", "fs"))             # фичестор
OUT = Path(os.environ.get("ECUP_OUT", "model_out"))
FS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

D0 = date(2025, 1, 1)
N_DAYS = 409
H = 30
NU = 250_000

VAL_ANCHOR = 378          # 2026-01-14 -> target 2026-01-15..2026-02-13 (LB-like)
HON_ANCHOR = 288          # 2025-10-16 -> target free of the selection rule
TRAIN_ANCHORS = [348, 334, 320, 306, 292, 278, 264, 250, 236, 222]   # all target windows end <= 378

def ds(i):
    return str(D0 + timedelta(days=int(i)))

def rmsle(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))

# --------------------------------------------------------------------- panels
_P = {}
def panel(name):
    if name not in _P:
        _P[name] = np.load(CACHE / f"P_{name}.npy")
    return _P[name]

_T = {}
def target(anchor):
    """30-day forward GMV. Memoised: calibration grids call this thousands of times."""
    if anchor not in _T:
        _T[anchor] = panel("gmv")[:, anchor + 1:anchor + H + 1].sum(axis=1, dtype=np.float64)
    return _T[anchor]

def cohort_rule(anchor):
    """users active in each of the 3 preceding 30-day blocks (replicates the organisers' rule)"""
    V = panel("visit")
    m = np.ones(NU, bool)
    for k in range(3):
        e = anchor - 30 * k
        s = e - 29
        m &= V[:, s:e + 1].sum(axis=1) > 0
    return m

# ---------------------------------------------------------------- feature store
WINDOWS = [7, 14, 30, 60, 90, 180, 365]
METRICS = [("gmv", np.float32), ("gmv_search", np.float32), ("ord", np.int32),
           ("cart", np.int32), ("srch", np.int32), ("visit", np.int32), ("engaged", np.int32)]

def _cumsum(M, dt):
    z = np.zeros((M.shape[0], 1), dt)
    return np.concatenate([z, M.astype(dt)], axis=1).cumsum(axis=1, dtype=dt)

def _last_nonzero_day(M, anchor):
    """day index of last positive value in [0, anchor]; -1 if none"""
    sub = M[:, :anchor + 1] > 0
    has = sub.any(axis=1)
    idx = anchor - np.argmax(sub[:, ::-1], axis=1)
    return np.where(has, idx, -1)

def build_features(anchors, verbose=True):
    """returns (dict anchor -> float32 (NU, F), list of feature names)"""
    names = []
    cols = {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    # ---- window sums per metric
    store = {}
    for mname, dt in METRICS:
        cs = _cumsum(panel(mname), np.float64 if dt == np.float32 else np.int64)
        for w in WINDOWS:
            for a in anchors:
                s = max(0, a - w + 1)
                store[(mname, w, a)] = cs[:, a + 1] - cs[:, s]
        for a in anchors:
            store[(mname, "all", a)] = cs[:, a + 1]
        # 30-day block lags. The end of a block can fall before day 0 for very early anchors —
        # without the guard the negative index wraps around and silently fabricates values.
        for k in range(1, 7):
            for a in anchors:
                e = a - 30 * (k - 1); s = e - 29
                store[(mname, f"blk{k}", a)] = (np.zeros(cs.shape[0], cs.dtype) if e < 0
                                                else cs[:, e + 1] - cs[:, max(0, s)])
        del cs
    # buy days
    BUY = (panel("gmv") > 0).astype(np.int8)
    csb = _cumsum(BUY, np.int64)
    for w in WINDOWS:
        for a in anchors:
            store[("buyday", w, a)] = csb[:, a + 1] - csb[:, max(0, a - w + 1)]
    for a in anchors:
        store[("buyday", "all", a)] = csb[:, a + 1]
    for k in range(1, 7):
        for a in anchors:
            e = a - 30 * (k - 1); s = e - 29
            store[("buyday", f"blk{k}", a)] = (np.zeros(csb.shape[0], csb.dtype) if e < 0
                                               else csb[:, e + 1] - csb[:, max(0, s)])
    del csb

    ALLM = [m for m, _ in METRICS] + ["buyday"]
    for m in ALLM:
        for w in WINDOWS + ["all"]:
            add(f"{m}_{w}", lambda a, m=m, w=w: store[(m, w, a)])
        for k in range(1, 7):
            add(f"{m}_blk{k}", lambda a, m=m, k=k: store[(m, f"blk{k}", a)])

    # ---- derived
    eps = 1e-6
    for w in [30, 90, 180, "all"]:
        add(f"gmv_cat_{w}", lambda a, w=w: store[("gmv", w, a)] - store[("gmv_search", w, a)])
        add(f"aov_{w}", lambda a, w=w: store[("gmv", w, a)] / np.maximum(store[("ord", w, a)], 1))
        add(f"gmv_per_visit_{w}", lambda a, w=w: store[("gmv", w, a)] / np.maximum(store[("visit", w, a)], 1))
        add(f"ord_per_cart_{w}", lambda a, w=w: store[("ord", w, a)] / np.maximum(store[("cart", w, a)], 1))
        add(f"cart_per_srch_{w}", lambda a, w=w: store[("cart", w, a)] / np.maximum(store[("srch", w, a)], 1))
        add(f"buyrate_{w}", lambda a, w=w: store[("buyday", w, a)] / np.maximum(store[("visit", w, a)], 1))
        add(f"srch_share_gmv_{w}", lambda a, w=w: store[("gmv_search", w, a)] / np.maximum(store[("gmv", w, a)], eps))
    # trend ratios
    add("gmv_30_over_90n", lambda a: store[("gmv", 30, a)] / (store[("gmv", 90, a)] / 3 + 1))
    add("gmv_30_over_180n", lambda a: store[("gmv", 30, a)] / (store[("gmv", 180, a)] / 6 + 1))
    add("gmv_7_over_30n", lambda a: store[("gmv", 7, a)] / (store[("gmv", 30, a)] / 30 * 7 + 1))
    add("gmv_blk1_over_blk2", lambda a: store[("gmv", "blk1", a)] / (store[("gmv", "blk2", a)] + 1))
    add("ord_30_over_90n", lambda a: store[("ord", 30, a)] / (store[("ord", 90, a)] / 3 + 1))
    add("visit_30_over_90n", lambda a: store[("visit", 30, a)] / (store[("visit", 90, a)] / 3 + 1))
    add("gmv_all_per_day", lambda a: store[("gmv", "all", a)] / (a + 1))
    add("gmv_life_rate30", lambda a: store[("gmv", "all", a)] / ((a + 1) / 30))

    # block volatility
    def blkstats(a, kind):
        B = np.stack([store[("gmv", f"blk{k}", a)] for k in range(1, 7)], axis=1)
        L = np.log1p(B)
        return L.std(axis=1) if kind == "std" else L.mean(axis=1)
    add("gmv_blk_logstd", lambda a: blkstats(a, "std"))
    add("gmv_blk_logmean", lambda a: blkstats(a, "mean"))

    # ---- recency & tenure
    G, V, C, S = panel("gmv"), panel("visit"), panel("cart"), panel("srch")
    rec_cache = {}
    def rec(a, M, key):
        k = (a, key)
        if k not in rec_cache:
            ln = _last_nonzero_day(M, a)
            rec_cache[k] = np.where(ln >= 0, a - ln, 999).astype(np.float32)
        return rec_cache[k]
    add("rec_visit", lambda a: rec(a, V, "v"))
    add("rec_buy", lambda a: rec(a, G, "g"))
    add("rec_cart", lambda a: rec(a, C, "c"))
    add("rec_srch", lambda a: rec(a, S, "s"))

    first_cache = {}
    def first(a):
        if a not in first_cache:
            sub = V[:, :a + 1] > 0
            has = sub.any(axis=1)
            first_cache[a] = np.where(has, np.argmax(sub, axis=1), a).astype(np.float32)
        return first_cache[a]
    add("tenure", lambda a: a - first(a))
    add("first_day", lambda a: first(a))

    # mean interpurchase gap + overdue ratio
    def meangap(a):
        lb = _last_nonzero_day(G, a)
        sub = G[:, :a + 1] > 0
        has = sub.any(axis=1)
        fb = np.where(has, np.argmax(sub, axis=1), 0)
        nb = store[("buyday", "all", a)]
        gap = np.where(nb > 1, (lb - fb) / np.maximum(nb - 1, 1), 999.0)
        return gap.astype(np.float32)
    gap_cache = {}
    def gapc(a):
        if a not in gap_cache: gap_cache[a] = meangap(a)
        return gap_cache[a]
    add("mean_gap", lambda a: gapc(a))
    add("overdue", lambda a: rec(a, G, "g") / np.maximum(gapc(a), 1))

    # peak daily gmv
    add("max_gmv_30", lambda a: G[:, max(0, a - 29):a + 1].max(axis=1))
    add("max_gmv_90", lambda a: G[:, max(0, a - 89):a + 1].max(axis=1))
    add("max_ord_30", lambda a: panel("ord")[:, max(0, a - 29):a + 1].max(axis=1))

    # ---- calendar of the TARGET window (critical for multi-anchor training)
    def tmid(a):
        return (D0 + timedelta(days=a + 15))
    add("tgt_doy_sin", lambda a: np.full(NU, np.sin(2 * np.pi * tmid(a).timetuple().tm_yday / 365.25)))
    add("tgt_doy_cos", lambda a: np.full(NU, np.cos(2 * np.pi * tmid(a).timetuple().tm_yday / 365.25)))
    add("tgt_month", lambda a: np.full(NU, tmid(a).month))
    add("hist_days", lambda a: np.full(NU, a + 1))

    X = {a: np.stack(cols[a], axis=1).astype(np.float32) for a in anchors}
    for a in anchors:
        np.nan_to_num(X[a], copy=False, posinf=0, neginf=0)
    if verbose:
        print(f"features: {len(names)} cols, anchors {anchors}, per-anchor {X[anchors[0]].nbytes/1e6:.0f} MB")
    return X, names

# ---------------------------------------------------------------- scoreboard
SB = OUT / "scoreboard.jsonl"
def record(name, family, val_rmsle, hon_rmsle=None, secs=None, notes="", extra=None):
    row = {"model": name, "family": family, "rmsle_val": round(float(val_rmsle), 5),
           "rmsle_hon": None if hon_rmsle is None else round(float(hon_rmsle), 5),
           "secs": None if secs is None else round(float(secs), 1), "notes": notes}
    if extra: row.update(extra)
    with open(SB, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    d = f" | honest {row['rmsle_hon']}" if row["rmsle_hon"] is not None else ""
    print(f"  {name:<52s} RMSLE {row['rmsle_val']:.5f}{d}  ({row['secs']}s)")
    return row

class Timer:
    def __enter__(self):
        self.t = time.time(); return self
    def __exit__(self, *a):
        self.dt = time.time() - self.t
