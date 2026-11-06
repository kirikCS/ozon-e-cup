"""Feature set v2 — the four groups the EDA flagged as untapped, plus pipelines from the
model catalogue that had only been used as standalone models.

  A. intra-week patterns              day-of-week profile, entropy, concentration, weekend share
  B. interpurchase interval structure moments, extremes, hazard-style ratios
  C. Search/Catalog at sequence level per-window channel splits and drift
  D. per-client funnel                personal conversion rates smoothed to the population
  E. catalogue pipelines              weekly lags and slopes, Croston/TSB rates, population ranks

Every window ends at the anchor, so nothing here can see the target.
"""
import numpy as np

from common import NU, N_DAYS, panel

DOW = ((np.arange(N_DAYS) + 2) % 7).astype(np.int64)      # 2025-01-01 was a Wednesday
ONEHOT = np.zeros((N_DAYS, 7), np.float32)
ONEHOT[np.arange(N_DAYS), DOW] = 1.0
EPS = 1e-6
AOV_PRIOR = 33.8

def _cs(M, dt):
    z = np.zeros((M.shape[0], 1), dt)
    return np.concatenate([z, M.astype(dt)], 1).cumsum(1, dtype=dt)

def _win(cs, a, b):
    """sum over days [a, b] inclusive, clamped to the available history"""
    b = min(b, N_DAYS - 1)
    if b < 0:
        return np.zeros(NU)
    a = max(a, 0)
    if a > b:
        return np.zeros(NU)
    return cs[:, b + 1] - cs[:, a]

def _gap_stats(B, a):
    """per-user statistics of gaps between consecutive True days in B[:, :a+1], fully vectorised"""
    r, c = np.nonzero(B[:, :a + 1])
    same = r[1:] == r[:-1]
    gr = r[1:][same]
    gp = (c[1:] - c[:-1])[same].astype(np.float64)
    n = np.bincount(gr, minlength=NU).astype(np.float64)
    s = np.bincount(gr, weights=gp, minlength=NU)
    ss = np.bincount(gr, weights=gp * gp, minlength=NU)
    mx = np.zeros(NU); np.maximum.at(mx, gr, gp)
    mn = np.full(NU, 1e9); np.minimum.at(mn, gr, gp); mn[n == 0] = 0.0
    last = np.zeros(NU)
    if len(gr):
        tail = np.append(gr[1:] != gr[:-1], True)          # last gap of each user
        last[gr[tail]] = gp[tail]
    nz = np.maximum(n, 1)
    mean = np.where(n > 0, s / nz, 999.0)
    var = np.where(n > 1, ss / nz - (s / nz) ** 2, 0.0)
    return dict(n=n, mean=mean, std=np.sqrt(np.maximum(var, 0)), max=mx, min=mn, last=last,
                gt30=np.bincount(gr, weights=(gp > 30).astype(np.float64), minlength=NU),
                gt60=np.bincount(gr, weights=(gp > 60).astype(np.float64), minlength=NU))

def _croston(G, alpha, sba, upto):
    z = np.zeros(NU); p = np.ones(NU); q = np.ones(NU); started = np.zeros(NU, bool)
    out = np.zeros((NU, upto + 1), np.float32)
    for t in range(upto + 1):
        y = G[:, t].astype(np.float64); nz = y > 0
        new = nz & ~started
        z[new] = y[new]; p[new] = q[new]
        upd = nz & started
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]
        started |= nz
        q[nz] = 1; q[~nz] += 1
        r = z / np.maximum(p, 1e-9)
        if sba:
            r = r * (1 - alpha / 2)
        out[:, t] = np.where(started, r, 0.0)
    return out

def _tsb(G, alpha, beta, upto):
    z = np.zeros(NU); pr = np.zeros(NU); started = np.zeros(NU, bool)
    out = np.zeros((NU, upto + 1), np.float32)
    for t in range(upto + 1):
        y = G[:, t].astype(np.float64); nz = y > 0
        z[nz & ~started] = y[nz & ~started]
        upd = nz & started
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        pr = np.where(nz, beta + (1 - beta) * pr, (1 - beta) * pr)
        started |= nz
        out[:, t] = np.where(started, pr * z, 0.0)
    return out

def build_v2(anchors, verbose=True):
    G, GS = panel("gmv"), panel("gmv_search")
    O, CA, SR = panel("ord"), panel("cart"), panel("srch")
    V, E = panel("visit"), panel("engaged")
    BUY = G > 0

    csG, csGS = _cs(G, np.float64), _cs(GS, np.float64)
    csO, csC, csS = _cs(O, np.int64), _cs(CA, np.int64), _cs(SR, np.int64)
    csV, csB, csE = _cs(V, np.int32), _cs(BUY, np.int32), _cs(E, np.int32)
    csOS = _cs((SR > 0) & (O > 0), np.int32)               # order days that also had search

    if verbose: print("  v2: Croston/TSB rates over the full timeline ...")
    RT = _tsb(G, 0.3, 0.02, N_DAYS - 1)
    RS = _croston(G, 0.3, True, N_DAYS - 1)

    gcache, vcache, fcache = {}, {}, {}
    def gs(a):
        if a not in gcache: gcache[a] = _gap_stats(BUY, a)
        return gcache[a]
    def vs(a):
        if a not in vcache: vcache[a] = _gap_stats(V > 0, a)
        return vcache[a]
    def first(a):
        if a not in fcache:
            sub = V[:, :a + 1] > 0
            fcache[a] = np.where(sub.any(1), np.argmax(sub, 1), a).astype(np.float64)
        return fcache[a]
    def rec_buy(a):
        sub = BUY[:, :a + 1]
        return np.where(sub.any(1), np.argmax(sub[:, ::-1], 1), 999).astype(np.float64)
    def dow_mat(M, a, W):
        s = max(0, a - W + 1)
        return M[:, s:a + 1].astype(np.float32) @ ONEHOT[s:a + 1]
    def weekly(a, cs, n):
        return np.stack([_win(cs, a - 7 * i - 6, a - 7 * i) for i in range(n)], 1)
    def conv(a, csn, csd, W, k=20.0):
        n = _win(csn, a - W + 1, a).astype(np.float64)
        d = _win(csd, a - W + 1, a).astype(np.float64)
        p = n.sum() / max(d.sum(), 1.0)
        return (n + k * p) / (d + k)

    names, cols = [], {a: [] for a in anchors}
    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.nan_to_num(np.asarray(fn(a), dtype=np.float32), posinf=0, neginf=0))

    # ---------------------------------------------------------- A. intra-week
    for W in (90, 180):
        for mn_, M in (("gmv", G), ("vis", V)):
            for k in range(7):
                add(f"dow_{mn_}_{W}_{k}", lambda a, M=M, W=W, k=k: (
                    lambda D: D[:, k] / np.maximum(D.sum(1), EPS))(dow_mat(M, a, W)))
            add(f"dow_{mn_}_{W}_entropy", lambda a, M=M, W=W: (
                lambda P: -(P * np.log(P + EPS)).sum(1))(
                    (lambda D: D / np.maximum(D.sum(1, keepdims=True), EPS))(dow_mat(M, a, W))))
            add(f"dow_{mn_}_{W}_top", lambda a, M=M, W=W: (
                lambda D: D.max(1) / np.maximum(D.sum(1), EPS))(dow_mat(M, a, W)))
            add(f"dow_{mn_}_{W}_weekend", lambda a, M=M, W=W: (
                lambda D: D[:, 5:7].sum(1) / np.maximum(D.sum(1), EPS))(dow_mat(M, a, W)))

    # ------------------------------------------------- B. interpurchase structure
    for k in ("n", "mean", "std", "max", "min", "last", "gt30", "gt60"):
        add(f"bgap_{k}", lambda a, k=k: gs(a)[k])
    add("bgap_cv", lambda a: gs(a)["std"] / np.maximum(gs(a)["mean"], EPS))
    add("bgap_last_over_mean", lambda a: gs(a)["last"] / np.maximum(gs(a)["mean"], EPS))
    add("bgap_range", lambda a: gs(a)["max"] - gs(a)["min"])
    add("rec_over_gapmean", lambda a: rec_buy(a) / np.maximum(gs(a)["mean"], EPS))
    add("rec_over_gapmax", lambda a: rec_buy(a) / np.maximum(gs(a)["max"], EPS))
    add("rec_gap_z", lambda a: (rec_buy(a) - gs(a)["mean"]) / np.maximum(gs(a)["std"], 1.0))
    for k in ("mean", "std", "max", "last", "gt30"):
        add(f"vgap_{k}", lambda a, k=k: vs(a)[k])

    # ---------------------------------------------------- C. channel structure
    for W in (30, 90, 365):
        add(f"chan_gmv_srch_{W}", lambda a, W=W:
            _win(csGS, a - W + 1, a) / np.maximum(_win(csG, a - W + 1, a), EPS))
        add(f"chan_ordday_srch_{W}", lambda a, W=W:
            _win(csOS, a - W + 1, a) / np.maximum(_win(csB, a - W + 1, a), EPS))
        add(f"chan_engaged_{W}", lambda a, W=W:
            _win(csE, a - W + 1, a) / np.maximum(_win(csV, a - W + 1, a), EPS))
    add("chan_srch_drift", lambda a:
        _win(csGS, a - 29, a) / np.maximum(_win(csG, a - 29, a), EPS)
        - _win(csGS, 0, a) / np.maximum(_win(csG, 0, a), EPS))

    # -------------------------------------------------------- D. per-client funnel
    for W in (30, 90, 365):
        add(f"conv_cart_per_srch_{W}", lambda a, W=W: conv(a, csC, csS, W))
        add(f"conv_ord_per_cart_{W}", lambda a, W=W: conv(a, csO, csC, W))
        add(f"conv_buyday_per_visit_{W}", lambda a, W=W: conv(a, csB, csV, W))
        add(f"basket_{W}", lambda a, W=W: conv(a, csO, csB, W, k=5.0))
        add(f"price_{W}", lambda a, W=W: (_win(csG, a - W + 1, a) + 5.0 * AOV_PRIOR) /
                                         (_win(csO, a - W + 1, a) + 5.0))
    add("conv_drift_30_365", lambda a:
        ((_win(csO, a - 29, a) + 20.0) / (_win(csC, a - 29, a) + 20.0)) /
        np.maximum((_win(csO, a - 364, a) + 20.0) / (_win(csC, a - 364, a) + 20.0), EPS))

    # ------------------------------------------------------ E. catalogue pipelines
    for i in range(12):
        add(f"wk_gmv_{i}", lambda a, i=i: _win(csG, a - 7 * i - 6, a - 7 * i))
    for i in range(8):
        add(f"wk_ord_{i}", lambda a, i=i: _win(csO, a - 7 * i - 6, a - 7 * i))
    def slope(a, cs, n):
        Wk = np.log1p(weekly(a, cs, n))
        x = np.arange(n, dtype=np.float64); x -= x.mean()
        return (Wk * x).sum(1) / (x * x).sum()
    for n in (12, 26):
        add(f"slope_gmv_{n}w", lambda a, n=n: slope(a, csG, n))
        add(f"slope_vis_{n}w", lambda a, n=n: slope(a, csV, n))
        add(f"wk_gmv_logstd_{n}", lambda a, n=n: np.log1p(weekly(a, csG, n)).std(1))
        add(f"wk_gmv_logmax_{n}", lambda a, n=n: np.log1p(weekly(a, csG, n)).max(1))
        add(f"wk_gmv_nonzero_{n}", lambda a, n=n: (weekly(a, csG, n) > 0).sum(1))
    add("tsb_rate", lambda a: RT[:, a])
    add("sba_rate", lambda a: RS[:, a])
    add("tsb_over_prev30", lambda a: 30 * RT[:, a] / np.maximum(_win(csG, a - 29, a), 1.0))

    def rank(v):
        return np.argsort(np.argsort(v)) / (NU - 1.0)
    add("rank_prev30", lambda a: rank(_win(csG, a - 29, a)))
    add("rank_prev365", lambda a: rank(_win(csG, a - 364, a)))
    add("rank_life_gmv", lambda a: rank(_win(csG, 0, a)))
    add("rank_buydays", lambda a: rank(_win(csB, 0, a)))
    add("rank_recency", lambda a: rank(rec_buy(a)))

    add("tenure_months", lambda a: (a - first(a)) / 30.0)
    add("buys_per_month", lambda a: _win(csB, 0, a) / np.maximum((a - first(a)) / 30.0, 1.0))
    add("gmv_per_month", lambda a: _win(csG, 0, a) / np.maximum((a - first(a)) / 30.0, 1.0))

    X = {a: np.stack(cols[a], 1).astype(np.float32) for a in anchors}
    if verbose:
        print(f"  v2: {len(names)} features, {X[anchors[0]].nbytes / 1e6:.0f} MB per anchor")
    return X, names
