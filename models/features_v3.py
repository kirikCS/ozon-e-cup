"""Feature set v3 — the catalog channel, which no earlier feature could see.

Every model in the stack is trained on the same 296 columns, and that is why nothing added since
stack9 has a correlation below 0.98 with the stack: a new architecture over identical inputs makes
its errors in the same places. This set is the first genuinely new information since the v2 round —
the search/catalog decomposition of carts and orders, which the panel prep never materialised.

Six groups:
  F  catalog volumes per window and their share of the funnel
  G  channel-specific conversion, smoothed to the population rate
  H  the mix of search-days and catalog-days, and its entropy
  I  drift: recent channel share against the older one
  J  recency and count of catalog events
  K  catalog AOV and its ratio to the search AOV
"""
import numpy as np

from common import NU, N_DAYS, panel

EPS = 1e-6
WINS = [7, 14, 30, 60, 90, 180]

def _cs(M, dt):
    z = np.zeros((M.shape[0], 1), dt)
    return np.concatenate([z, M.astype(dt)], 1).cumsum(1, dtype=dt)

def _win(cs, a, b):
    b = min(b, N_DAYS - 1)
    if b < 0:
        return np.zeros(cs.shape[0])
    a = max(a, 0)
    if a > b:
        return np.zeros(cs.shape[0])
    return cs[:, b + 1] - cs[:, a]

def _recency(B, a, cap=400):
    """days since the last True on or before day a; cap when it never happened"""
    idx = np.full(B.shape[0], -1, np.int32)
    sub = B[:, :a + 1]
    has = sub.any(1)
    idx[has] = sub.shape[1] - 1 - np.argmax(sub[has, ::-1], axis=1)
    out = np.full(B.shape[0], float(cap), np.float32)
    out[has] = a - idx[has]
    return out

def build(anchors, verbose=True):
    names, cols = [], {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    CC, CO = panel("catcart"), panel("catord")
    GC = panel("gmvcat")
    CF, SF = panel("catflag"), panel("srchflag")
    TC, TO = panel("cart"), panel("ord")
    G, GS = panel("gmv"), panel("gmv_search")

    csCC, csCO = _cs(CC, np.int64), _cs(CO, np.int64)
    csGC = _cs(GC, np.float64)
    csCF, csSF = _cs(CF, np.int64), _cs(SF, np.int64)
    csTC, csTO = _cs(TC, np.int64), _cs(TO, np.int64)
    csG, csGS = _cs(G, np.float64), _cs(GS, np.float64)
    csCOD = _cs((CO > 0).astype(np.int8), np.int64)         # days with a catalog order

    # population conversion rates, for the empirical-Bayes smoothing in group G
    a0 = anchors[0]
    P_CAT = float(_win(csCO, 0, a0).sum() / max(_win(csCC, 0, a0).sum(), 1))
    P_SRC = float((_win(csTO, 0, a0).sum() - _win(csCO, 0, a0).sum())
                  / max(_win(csTC, 0, a0).sum() - _win(csCC, 0, a0).sum(), 1))

    # ---- F: catalog volumes and funnel share
    for w in WINS:
        add(f"catcart_{w}", lambda a, w=w: _win(csCC, a - w + 1, a))
        add(f"catord_{w}", lambda a, w=w: _win(csCO, a - w + 1, a))
        add(f"gmvcat_{w}", lambda a, w=w: _win(csGC, a - w + 1, a))
        add(f"catcart_share_{w}", lambda a, w=w:
            _win(csCC, a - w + 1, a) / np.maximum(_win(csTC, a - w + 1, a), EPS))
        add(f"catord_share_{w}", lambda a, w=w:
            _win(csCO, a - w + 1, a) / np.maximum(_win(csTO, a - w + 1, a), EPS))
        add(f"gmvcat_share_{w}", lambda a, w=w:
            _win(csGC, a - w + 1, a) / np.maximum(_win(csG, a - w + 1, a), EPS))
    add("catcart_all", lambda a: _win(csCC, 0, a))
    add("catord_all", lambda a: _win(csCO, 0, a))
    add("gmvcat_all", lambda a: _win(csGC, 0, a))
    add("catord_share_all", lambda a: _win(csCO, 0, a) / np.maximum(_win(csTO, 0, a), EPS))
    add("gmvcat_share_all", lambda a: _win(csGC, 0, a) / np.maximum(_win(csG, 0, a), EPS))

    # ---- G: channel conversion, shrunk to the population rate (prior weight 20 carts)
    for w in (90, 180, "all"):
        def cv_cat(a, w=w):
            lo = 0 if w == "all" else a - w + 1
            c, o = _win(csCC, lo, a), _win(csCO, lo, a)
            return (o + 20 * P_CAT) / (c + 20)
        def cv_src(a, w=w):
            lo = 0 if w == "all" else a - w + 1
            c = _win(csTC, lo, a) - _win(csCC, lo, a)
            o = _win(csTO, lo, a) - _win(csCO, lo, a)
            return (o + 20 * P_SRC) / (c + 20)
        add(f"conv_cat_{w}", cv_cat)
        add(f"conv_src_{w}", cv_src)
        add(f"conv_gap_{w}", lambda a, w=w, f=cv_cat, g=cv_src: f(a) - g(a))

    # ---- H: day mix between the two entry points
    for w in (30, 90, 180):
        add(f"catdays_{w}", lambda a, w=w: _win(csCF, a - w + 1, a))
        add(f"srchdays_{w}", lambda a, w=w: _win(csSF, a - w + 1, a))
        def mix(a, w=w):
            c, s = _win(csCF, a - w + 1, a), _win(csSF, a - w + 1, a)
            return c / np.maximum(c + s, EPS)
        add(f"catday_share_{w}", mix)
        def ent(a, w=w, m=mix):
            p = np.clip(m(a), 1e-6, 1 - 1e-6)
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))
        add(f"channel_entropy_{w}", ent)

    # ---- I: drift of the catalog share, recent against older
    for short, long in ((30, 180), (60, 180), (30, 90)):
        add(f"catord_share_drift_{short}_{long}", lambda a, s=short, l=long:
            _win(csCO, a - s + 1, a) / np.maximum(_win(csTO, a - s + 1, a), EPS)
            - _win(csCO, a - l + 1, a) / np.maximum(_win(csTO, a - l + 1, a), EPS))
        add(f"gmvcat_ratio_{short}_{long}", lambda a, s=short, l=long:
            (_win(csGC, a - s + 1, a) * (l / s)) / np.maximum(_win(csGC, a - l + 1, a), 1.0))

    # ---- J: recency and totals
    add("rec_catord", lambda a: _recency(CO > 0, a))
    add("rec_catcart", lambda a: _recency(CC > 0, a))
    add("rec_catvisit", lambda a: _recency(CF > 0, a))
    add("catorddays_all", lambda a: _win(csCOD, 0, a))
    add("catord_per_month", lambda a: _win(csCO, 0, a) / np.maximum((a + 1) / 30.0, 1.0))
    add("ever_catord", lambda a: (_win(csCO, 0, a) > 0).astype(np.float32))

    # ---- K: catalog basket size against the search basket
    for w in (90, 180, "all"):
        def aov_cat(a, w=w):
            lo = 0 if w == "all" else a - w + 1
            return _win(csGC, lo, a) / np.maximum(_win(csCO, lo, a), 1)
        def aov_src(a, w=w):
            lo = 0 if w == "all" else a - w + 1
            return _win(csGS, lo, a) / np.maximum(_win(csTO, lo, a) - _win(csCO, lo, a), 1)
        add(f"aov_cat_{w}", aov_cat)
        add(f"aov_src_{w}", aov_src)
        add(f"aov_ratio_{w}", lambda a, f=aov_cat, g=aov_src: f(a) / np.maximum(g(a), EPS))

    X = {a: np.stack(cols[a], 1).astype(np.float32) for a in anchors}
    if verbose:
        print(f"  v3: {len(names)} features, {X[anchors[0]].nbytes / 1e6:.0f} MB per anchor")
    return X, names
