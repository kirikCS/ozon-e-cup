"""Feature extension groups — the four EDA areas the 165-column store never touched.

Each group is built strictly from days [0, anchor]; no target-window information can enter
by construction. Groups return (names, {anchor: float32 (NU, k)}) and are concatenated
AFTER the base columns, so the base matrix stays bit-identical to the frozen store.

  gap     inter-purchase interval structure: std, cv, last, max, recent-3 trend,
          burstiness, open-interval ratios; plus visit-gap regularity
  dow     per-client weekday profile: weekend share, entropy, concentration
          (target is a 30-day sum covering every weekday ~4x, so this is a behavioural
          signature, not calendar alignment)
  chan    search vs catalogue split of the funnel at panel level — needs fx_panels.py
  funnel  personal conversion rates search->cart->order per channel — needs fx_panels.py
  noiseK  K iid N(0,1) columns — control for the column-count/colsample dilution effect

Missing-value sentinels follow the base store where a convention exists (recency/gap 999),
otherwise -1 (all real values are >= 0) and -2 for burstiness (real range (-1, 1)).
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, NU, D0

WD0 = date(2025, 1, 1).weekday()          # 2 = Wednesday


# ------------------------------------------------------------------ gap machinery
def _gap_snapshots(M, anchors):
    """Single pass over days, snapshotting per-user gap statistics at each anchor.
    M: (NU, T) activity matrix (>0 = active day). Gap = days between consecutive active days."""
    wset = set(int(a) for a in anchors)
    maxa = max(wset)
    MT = np.ascontiguousarray((M[:, :maxa + 1] > 0).T)
    last = np.full(NU, -1, np.int64)
    n = np.zeros(NU, np.int64)
    s1 = np.zeros(NU, np.float64)
    s2 = np.zeros(NU, np.float64)
    mx = np.zeros(NU, np.float64)
    g1 = np.zeros(NU, np.float64)
    g2 = np.zeros(NU, np.float64)
    g3 = np.zeros(NU, np.float64)
    snaps = {}
    for d in range(maxa + 1):
        idx = np.flatnonzero(MT[d])
        if idx.size:
            l = last[idx]
            m = l >= 0
            if m.any():
                i2 = idx[m]
                g = (d - l[m]).astype(np.float64)
                n[i2] += 1
                s1[i2] += g
                s2[i2] += g * g
                mx[i2] = np.maximum(mx[i2], g)
                g3[i2] = g2[i2]
                g2[i2] = g1[i2]
                g1[i2] = g
            last[idx] = d
        if d in wset:
            snaps[d] = _gap_features(d, last, n, s1, s2, mx, g1, g2, g3)
    return snaps


def _gap_features(a, last, n, s1, s2, mx, g1, g2, g3):
    """returns dict name -> float32 (order fixed by insertion)"""
    out = {}
    has1 = n >= 1                                     # at least one completed gap
    has2 = n >= 2
    has3 = n >= 3
    mean = np.divide(s1, n, out=np.ones_like(s1), where=has1)
    var = np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0.0)
    std = np.sqrt(var)
    out["gap_std"] = np.where(has2, std, -1.0)
    out["gap_cv"] = np.where(has2, std / np.maximum(mean, 1e-9), -1.0)
    out["gap_last"] = np.where(has1, g1, 999.0)
    out["gap_max"] = np.where(has1, mx, 999.0)
    rec3 = (g1 + g2 + g3) / 3.0
    out["gap_recent3"] = np.where(has3, rec3, 999.0)
    out["gap_trend"] = np.where(has3, rec3 / np.maximum(mean, 1e-9), -1.0)
    out["gap_burst"] = np.where(has2, (std - mean) / np.maximum(std + mean, 1e-9), -2.0)
    open_ = a - last
    out["open_over_max"] = np.where(has1, open_ / np.maximum(mx, 1.0), -1.0)
    out["open_over_last"] = np.where(has1, open_ / np.maximum(g1, 1.0), -1.0)
    return {k: v.astype(np.float32) for k, v in out.items()}


def _build_gap(anchors):
    buy = _gap_snapshots(panel("gmv"), anchors)
    vis = _gap_snapshots(panel("visit"), anchors)
    b_keys = ["gap_std", "gap_cv", "gap_last", "gap_max", "gap_recent3",
              "gap_trend", "gap_burst", "open_over_max", "open_over_last"]
    v_keys = ["gap_std", "gap_cv", "gap_last", "gap_max", "open_over_max"]
    names = [f"b_{k}" for k in b_keys] + [f"v_{k}" for k in v_keys]
    X = {a: np.stack([buy[a][k] for k in b_keys] + [vis[a][k] for k in v_keys], axis=1)
         for a in anchors}
    return names, X


# ------------------------------------------------------------------ weekday profile
def _build_dow(anchors):
    names, cols = [], {a: [] for a in anchors}
    wd_all = (np.arange(max(anchors) + 1) + WD0) % 7

    def profile(M, a, w):
        s = 0 if w == "all" else max(0, a - w + 1)
        sub = M[:, s:a + 1]
        W = np.zeros((sub.shape[1], 7), np.float32)
        W[np.arange(sub.shape[1]), wd_all[s:a + 1]] = 1.0
        S = sub.astype(np.float32) @ W                    # (NU, 7) weekday sums
        tot = S.sum(axis=1)
        ok = tot > 0
        p = S / np.maximum(tot, 1e-9)[:, None]
        weekend = np.where(ok, p[:, 5] + p[:, 6], -1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1) / np.log(7.0)
        ent = np.where(ok, ent, -1.0)
        top = np.where(ok, p.max(axis=1), -1.0)
        return weekend.astype(np.float32), ent.astype(np.float32), top.astype(np.float32)

    for mname in ["gmv", "visit"]:
        M = panel(mname)
        for w in [90, "all"]:
            per_a = {a: profile(M, a, w) for a in anchors}
            for j, stat in enumerate(["wknd", "ent", "top"]):
                names.append(f"dow_{stat}_{mname}_{w}")
                for a in anchors:
                    cols[a].append(per_a[a][j])
    X = {a: np.stack(cols[a], axis=1) for a in anchors}
    return names, X


# ------------------------------------------------------------------ channel split
def _cums(M, dt=np.int64):
    z = np.zeros((M.shape[0], 1), dt)
    return np.concatenate([z, M.astype(dt)], axis=1).cumsum(axis=1, dtype=dt)


def _wsum(cs, a, w):
    s = 0 if w == "all" else max(0, a - w + 1)
    return (cs[:, a + 1] - cs[:, s]).astype(np.float64)


def _recency(M, a):
    sub = M[:, :a + 1] > 0
    has = sub.any(axis=1)
    idx = a - np.argmax(sub[:, ::-1], axis=1)
    return np.where(has, (a - idx).astype(np.float32), 999.0).astype(np.float32)


def _channel_sums(anchors):
    """shared between chan and funnel: window sums of the channel panels.
    Cumsums are int64 (NU, 410) = 0.8 GB each, so build -> extract -> free one at a time."""
    P_sord, P_scart = panel("sord"), panel("scart")
    P_ord, P_cart = panel("ord"), panel("cart")
    P_cord = np.maximum(P_ord.astype(np.int32) - P_sord, 0)
    P_ccart = np.maximum(P_cart.astype(np.int32) - P_scart, 0)
    S = {}
    for k, p in [("sord", P_sord), ("scart", P_scart), ("cord", P_cord), ("ccart", P_ccart),
                 ("sday", panel("sday")), ("catday", panel("catday")),
                 ("ord", P_ord), ("cart", P_cart), ("srch", panel("srch"))]:
        cs = _cums(p)
        for w in [30, 90, "all"]:
            for a in anchors:
                S[(k, w, a)] = _wsum(cs, a, w)
        del cs
    del P_ccart
    return S, P_cord, None


def _build_chan(anchors, shared=None):
    S, P_cord, _ = shared if shared else _channel_sums(anchors)
    names, cols = [], {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    for k in ["sord", "cord", "scart", "ccart", "sday", "catday"]:
        for w in [30, 90, "all"]:
            add(f"{k}_{w}", lambda a, k=k, w=w: S[(k, w, a)])
    for w in [30, 90, "all"]:
        add(f"ord_srch_share_{w}",
            lambda a, w=w: S[("sord", w, a)] / np.maximum(S[("ord", w, a)], 1))
        add(f"cart_srch_share_{w}",
            lambda a, w=w: S[("scart", w, a)] / np.maximum(S[("cart", w, a)], 1))
        add(f"day_srch_share_{w}",
            lambda a, w=w: S[("sday", w, a)] / np.maximum(S[("sday", w, a)] + S[("catday", w, a)], 1))
    add("ord_srch_share_shift",
        lambda a: S[("sord", 30, a)] / np.maximum(S[("ord", 30, a)], 1)
                - S[("sord", "all", a)] / np.maximum(S[("ord", "all", a)], 1))
    add("day_srch_share_shift",
        lambda a: S[("sday", 30, a)] / np.maximum(S[("sday", 30, a)] + S[("catday", 30, a)], 1)
                - S[("sday", "all", a)] / np.maximum(S[("sday", "all", a)] + S[("catday", "all", a)], 1))
    rec_specs = [("rec_sord", "sord"), ("rec_sday", "sday"), ("rec_catday", "catday")]
    for nm, pn in rec_specs:
        add(nm, lambda a, pn=pn: _recency(panel(pn), a))
    add("rec_cord", lambda a: _recency(P_cord, a))
    X = {a: np.stack(cols[a], axis=1) for a in anchors}
    return names, X


def _build_funnel(anchors, shared=None):
    S, _, _ = shared if shared else _channel_sums(anchors)
    names, cols = [], {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    for w in [30, 90, "all"]:
        add(f"s2c_{w}", lambda a, w=w: S[("scart", w, a)] / np.maximum(S[("srch", w, a)], 1))
        add(f"s2o_{w}", lambda a, w=w: S[("sord", w, a)] / np.maximum(S[("srch", w, a)], 1))
        add(f"sc2o_{w}", lambda a, w=w: S[("sord", w, a)] / np.maximum(S[("scart", w, a)], 1))
        add(f"cday2c_{w}", lambda a, w=w: S[("ccart", w, a)] / np.maximum(S[("catday", w, a)], 1))
        add(f"cc2o_{w}", lambda a, w=w: S[("cord", w, a)] / np.maximum(S[("ccart", w, a)], 1))
        add(f"cart_unconv_{w}", lambda a, w=w: S[("cart", w, a)] - S[("ord", w, a)])
    X = {a: np.stack(cols[a], axis=1) for a in anchors}
    return names, X


# ------------------------------------------------------------------ holiday propensity
# The competition target window (2026-02-14..03-15) contains Feb 23 and Mar 8 — the two
# big gifting holidays. History contains their 2025 editions (days 48-71 from D0) and the
# exact year-ago slice of the test window. Per-user gift-response traits are computable at
# every anchor >= 75 and are a feature class no prior screen touched. The SELECT target
# window contains New Year, so gifting propensity is locally expressible.
_HOL = {
    "f23": (48, 58),      # 2025-02-18..02-28  (Feb 23 +-5)
    "m8": (61, 71),       # 2025-03-03..03-13  (Mar 8 +-5)
    "yago": (44, 73),     # 2025-02-14..03-15  (year-ago test window)
    "nov11": (309, 318),  # 2025-11-06..11-15  (Nov 11 sale +-4)
    "ny": (348, 371),     # 2025-12-15..2026-01-07
}


def _build_holiday(anchors):
    G = panel("gmv")
    BUY = (G > 0)
    names, cols = [], {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    def wsum(M, s, e, a, dt=np.float64):
        if e > a:                     # window must be FULLY inside history — no partial slices
            return None
        return M[:, s:e + 1].sum(axis=1, dtype=dt)

    def lift(key, a):
        s, e = _HOL[key]
        g = wsum(G, s, e, a)
        if g is None:
            return np.full(NU, -1.0)
        rate = G[:, :a + 1].sum(axis=1, dtype=np.float64) / (a + 1)
        return g / (rate * (min(e, a) - s + 1) + 1.0)

    for key in ["f23", "m8", "nov11", "ny"]:
        add(f"hol_{key}_lift", lambda a, key=key: lift(key, a))
    add("hol_gift_lift", lambda a: np.where(lift("f23", a) >= 0,
                                            lift("f23", a) + lift("m8", a), -2.0))
    for key, m, nm in [("yago", G, "gmv"), ("yago", BUY, "bdays")]:
        add(f"hol_yago_{nm}", lambda a, m=m: np.log1p(
            wsum(m, *_HOL["yago"], a) if wsum(m, *_HOL["yago"], a) is not None
            else np.zeros(NU)))
    add("hol_yago_share", lambda a: (wsum(G, *_HOL["yago"], a) /
                                     (G[:, :a + 1].sum(axis=1, dtype=np.float64) + 1e-6))
        if wsum(G, *_HOL["yago"], a) is not None else np.full(NU, -1.0))
    X = {a: np.stack(cols[a], axis=1) for a in anchors}
    return names, X


# ------------------------------------------------------------------ cell-mean encoding
CM_QBINS = 20          # quantile bins per dimension (oracle_floor's grid: 21 buckets)
CM_PAST = 5            # past anchors pooled, spaced 28d, most recent ends at a-30


def _cm_feats(a):
    """the 5 oracle-grid columns, computed directly from panels (cheap, no full store)"""
    G = panel("gmv")
    gmv_30 = G[:, max(0, a - 29):a + 1].sum(axis=1, dtype=np.float64)
    gmv_90 = G[:, max(0, a - 89):a + 1].sum(axis=1, dtype=np.float64)
    BUY = G[:, :a + 1] > 0
    buyday_all = BUY.sum(axis=1, dtype=np.int64)
    buyday_30 = BUY[:, max(0, a - 29):].sum(axis=1, dtype=np.int64)
    has = BUY.any(axis=1)
    last = a - np.argmax(BUY[:, ::-1], axis=1)
    rec_buy = np.where(has, (a - last).astype(np.float64), 999.0)
    return [buyday_all.astype(np.float64), gmv_90, buyday_30.astype(np.float64),
            rec_buy, gmv_30]


def _cm_key(a):
    cols = _cm_feats(a)
    q = []
    for v in cols:
        qs = np.percentile(v, np.linspace(100 / (CM_QBINS + 1), 100 - 100 / (CM_QBINS + 1),
                                          CM_QBINS))
        q.append(np.searchsorted(qs, v).astype(np.int64))
    key = q[0]
    for c in q[1:]:
        key = key * (CM_QBINS + 1) + c
    return key


def _build_cellmean(anchors):
    """Cross-user pooling: EB-shrunk mean log1p(target) of the user's oracle-grid cell,
    accumulated over CM_PAST past anchors whose target windows are fully observable at a
    (t' + 30 <= a). Injects neighbour-target information no within-user aggregate has."""
    from common import target as _tgt
    names = ["cm_mean_k20", "cm_mean_k5", "cm_zero_k20", "cm_logn"]
    X = {}
    for a in anchors:
        past = [a - 30 - 28 * i for i in range(CM_PAST)]
        past = [t for t in past if t >= 64]
        stats = {}
        tot_s, tot_n, tot_z = 0.0, 0, 0
        for t in past:
            k = _cm_key(t)
            ly = np.log1p(_tgt(t))
            nc = k.max() + 1
            n = np.bincount(k, minlength=nc)
            s = np.bincount(k, weights=ly, minlength=nc)
            z = np.bincount(k, weights=(ly == 0).astype(np.float64), minlength=nc)
            stats[t] = (n, s, z)
            tot_s += ly.sum(); tot_n += NU; tot_z += int((ly == 0).sum())
        pop = tot_s / tot_n
        zpop = tot_z / tot_n
        ka = _cm_key(a)
        nc_a = ka.max() + 1
        n_acc = np.zeros(nc_a); s_acc = np.zeros(nc_a); z_acc = np.zeros(nc_a)
        for t in past:
            n, s, z = stats[t]
            m = min(len(n), nc_a)
            n_acc[:m] += n[:m]; s_acc[:m] += s[:m]; z_acc[:m] += z[:m]
        nk = n_acc[ka]; sk = s_acc[ka]; zk = z_acc[ka]
        cm20 = (sk + 20.0 * pop) / (nk + 20.0)
        cm5 = (sk + 5.0 * pop) / (nk + 5.0)
        cz = (zk + 20.0 * zpop) / (nk + 20.0)
        X[a] = np.stack([cm20, cm5, cz, np.log1p(nk)], axis=1).astype(np.float32)
    return names, X


# ------------------------------------------------------------------ dispatch
def build_group(spec, anchors):
    """spec: 'gap' | 'dow' | 'chan' | 'funnel' | 'noiseK' | 'a+b+...' (union, in order)"""
    anchors = [int(a) for a in anchors]
    if "+" in spec:
        names, Xs = [], {a: [] for a in anchors}
        shared = _channel_sums(anchors) if ("chan" in spec or "funnel" in spec) else None
        for part in spec.split("+"):
            n, X = _dispatch(part, anchors, shared)
            names += [f"{part}:{x}" for x in n]
            for a in anchors:
                Xs[a].append(X[a])
        return names, {a: np.concatenate(Xs[a], axis=1) for a in anchors}
    n, X = _dispatch(spec, anchors, None)
    return [f"{spec}:{x}" for x in n], X


def _dispatch(part, anchors, shared):
    if part == "gap":
        return _build_gap(anchors)
    if part == "dow":
        return _build_dow(anchors)
    if part == "chan":
        return _build_chan(anchors, shared)
    if part == "funnel":
        return _build_funnel(anchors, shared)
    if part == "cellmean":
        return _build_cellmean(anchors)
    if part == "holiday":
        return _build_holiday(anchors)
    if part.startswith("noise"):
        # noise16 / noise16@2 — @D picks an independent draw, so the dilution floor can be
        # measured across draws, not conditional on a single noise realisation
        body, _, draw = part[5:].partition("@")
        k = int(body or 16)
        d = int(draw or 0)
        names = [f"n{i}" for i in range(k)]
        X = {a: np.random.default_rng(9000 + a + 1_000_000 * d)
                .standard_normal((NU, k)).astype(np.float32) for a in anchors}
        return names, X
    raise ValueError(f"unknown group {part}")
