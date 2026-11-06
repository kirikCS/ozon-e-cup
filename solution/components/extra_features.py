"""Extended feature groups from text.md §10 — additions on top of build_features().

Four untested groups per text.md:
  1. Intra-week patterns at client level (day-of-week GMV, visit fractions)
  2. Purchase interval distribution (std/CV of inter-purchase gaps)
  3. Search/Catalog channel trend over time (srch_share shift)
  4. Personal funnel co-occurrence rates (srch+cart+ord same day)

Usage:
    from extra_features import build_extra_features
    X_extra, extra_names = build_extra_features(anchors)
    # Stack with base features: X_all = np.concatenate([X_base, X_extra], axis=1)
"""
import numpy as np
from datetime import date
from common import panel, NU, WINDOWS

WD0 = date(2025, 1, 1).weekday()   # Wednesday = 2; day_d has weekday (d+WD0)%7


def build_extra_features(anchors, verbose=True):
    """Returns (dict anchor -> float32 (NU, F_extra)), list of extra names."""
    names = []
    cols  = {a: [] for a in anchors}

    def add(name, fn):
        names.append(name)
        for a in anchors:
            cols[a].append(np.asarray(fn(a), dtype=np.float32))

    G = panel("gmv")
    S = panel("srch")
    CA = panel("cart")
    OR = panel("ord")
    V  = panel("visit")

    # ── 1. Day-of-week GMV fractions (last 90 days) ──────────────────────────
    # weekend_gmv_share: fraction of GMV on Sat (5) or Sun (6)
    def wd_mask(a, w, *weekdays):
        """Boolean mask for days in the window [a-w+1, a] on given weekdays."""
        s = max(0, a - w + 1)
        # days in window
        all_days = np.arange(s, a + 1)
        wd_arr   = (all_days + WD0) % 7
        include  = np.isin(wd_arr, weekdays)
        sel_days = all_days[include]
        return sel_days

    for w in [30, 90]:
        def weekend_share(a, w=w):
            sel = wd_mask(a, w, 5, 6)  # Sat, Sun
            if len(sel) == 0:
                return np.zeros(NU, np.float32)
            num = G[:, sel].sum(1).astype(np.float64)
            den = G[:, max(0, a-w+1):a+1].sum(1).astype(np.float64)
            return (num / np.maximum(den, 1)).astype(np.float32)
        add(f"weekend_gmv_share_{w}", weekend_share)

        def weekday_share(a, w=w):
            sel = wd_mask(a, w, 0, 1, 2, 3, 4)  # Mon-Fri
            if len(sel) == 0:
                return np.zeros(NU, np.float32)
            num = G[:, sel].sum(1).astype(np.float64)
            den = G[:, max(0, a-w+1):a+1].sum(1).astype(np.float64)
            return (num / np.maximum(den, 1)).astype(np.float32)
        add(f"weekday_gmv_share_{w}", weekday_share)

    # weekend vs weekday buy-day ratio (regularity on weekends)
    for w in [30, 90]:
        def weekend_visit_rate(a, w=w):
            sel_wk = wd_mask(a, w, 5, 6)
            if len(sel_wk) == 0:
                return np.zeros(NU, np.float32)
            n_wk = G[:, sel_wk] > 0
            sel_wd = wd_mask(a, w, 0, 1, 2, 3, 4)
            n_wd = G[:, sel_wd] > 0 if len(sel_wd) > 0 else np.zeros((NU, 1), bool)
            rate_wk = n_wk.sum(1) / max(len(sel_wk), 1)
            rate_wd = n_wd.sum(1) / max(len(sel_wd), 1)
            return (rate_wk / np.maximum(rate_wd, 0.01)).astype(np.float32)
        add(f"weekend_over_weekday_buyrate_{w}", weekend_visit_rate)

    # ── 2. Inter-purchase gap statistics ─────────────────────────────────────
    # Approximate std of inter-purchase gaps from buy-day positions.
    # Using block-level buy counts to estimate gap variance cheaply.
    def gap_cv(a):
        """Coefficient of variation of monthly buy-block counts.

        A user with highly variable monthly activity (high CV) tends to have
        irregular inter-purchase gaps. CV = std(blk_counts) / mean(blk_counts).
        """
        blocks = []
        for k in range(1, 7):
            e = a - 30 * (k - 1)
            s = e - 29
            if s < 0:
                break
            bd = (G[:, s:e+1] > 0).sum(1).astype(np.float64)
            blocks.append(bd)
        if len(blocks) < 2:
            return np.zeros(NU, np.float32)
        B = np.stack(blocks, axis=1)  # (NU, K)
        mu  = B.mean(1)
        std = B.std(1)
        return (std / np.maximum(mu, 0.1)).astype(np.float32)

    add("buyday_monthly_cv", gap_cv)

    # trend in inter-purchase frequency: recent blocks vs older blocks
    def gap_trend(a):
        """Ratio of buy-days in last 30d vs avg of 30-60d and 60-90d."""
        e0 = a;    s0 = max(0, e0 - 29)
        e1 = a-30; s1 = max(0, e1 - 29)
        e2 = a-60; s2 = max(0, e2 - 29)
        if e2 < 0:
            return np.zeros(NU, np.float32)
        b0 = (G[:, s0:e0+1] > 0).sum(1).astype(np.float64)
        b1 = (G[:, s1:e1+1] > 0).sum(1).astype(np.float64)
        b2 = (G[:, s2:e2+1] > 0).sum(1).astype(np.float64)
        avg_old = (b1 + b2) / 2.0
        return (b0 / np.maximum(avg_old, 0.1)).astype(np.float32)

    add("buyday_freq_trend", gap_trend)

    # ── 3. Search/Catalog channel trend ──────────────────────────────────────
    # Already have srch_share_gmv_{w} for static windows.
    # New: how is the share SHIFTING?
    GS = panel("gmv_search")  # guaranteed to exist (in METRICS)

    def srch_share_shift(a):
        """Recent 30d srch_share vs 90d srch_share — channel shift signal."""
        G30 = G[:, max(0, a-29):a+1].sum(1).astype(np.float64)
        S30 = GS[:, max(0, a-29):a+1].sum(1).astype(np.float64)
        G90 = G[:, max(0, a-89):a+1].sum(1).astype(np.float64)
        S90 = GS[:, max(0, a-89):a+1].sum(1).astype(np.float64)
        share30 = S30 / np.maximum(G30, 1)
        share90 = S90 / np.maximum(G90, 1)
        return (share30 - share90).astype(np.float32)  # positive = shifting to search

    def srch_share_ratio(a):
        G30 = G[:, max(0, a-29):a+1].sum(1).astype(np.float64)
        S30 = GS[:, max(0, a-29):a+1].sum(1).astype(np.float64)
        G90 = G[:, max(0, a-89):a+1].sum(1).astype(np.float64)
        S90 = GS[:, max(0, a-89):a+1].sum(1).astype(np.float64)
        share30 = S30 / np.maximum(G30, 1)
        share90 = S90 / np.maximum(G90, 1)
        return (share30 / np.maximum(share90, 0.01)).astype(np.float32)

    add("srch_share_shift_30v90", srch_share_shift)
    add("srch_share_ratio_30v90", srch_share_ratio)

    def srch_share_blk_trend(a):
        results = []
        for k in range(1, 4):
            e = a - 30 * (k - 1); s = max(0, e - 29)
            gblk = G[:, s:e+1].sum(1).astype(np.float64)
            sblk = GS[:, s:e+1].sum(1).astype(np.float64)
            results.append(sblk / np.maximum(gblk, 1))
        if len(results) < 3:
            return np.zeros(NU, np.float32)
        return (results[0] - results[2]).astype(np.float32)

    add("srch_share_blk1v3_trend", srch_share_blk_trend)

    # ── 4. Personal funnel co-occurrence rates ────────────────────────────────
    # Days where search AND cart happened on the same day
    def srch_cart_cooccur(a, w=30):
        """Fraction of search days that also had a cart event (same day)."""
        s = max(0, a - w + 1)
        has_srch  = S[:, s:a+1] > 0
        has_cart  = CA[:, s:a+1] > 0
        n_srch    = has_srch.sum(1).astype(np.float64)
        n_both    = (has_srch & has_cart).sum(1).astype(np.float64)
        return (n_both / np.maximum(n_srch, 1)).astype(np.float32)

    def cart_ord_cooccur(a, w=30):
        """Fraction of cart days that also had an order event (same day)."""
        s = max(0, a - w + 1)
        has_cart  = CA[:, s:a+1] > 0
        has_ord   = OR[:, s:a+1] > 0
        n_cart    = has_cart.sum(1).astype(np.float64)
        n_both    = (has_cart & has_ord).sum(1).astype(np.float64)
        return (n_both / np.maximum(n_cart, 1)).astype(np.float32)

    def triple_cooccur(a, w=30):
        """Fraction of days with search AND cart AND order (full funnel completion)."""
        s = max(0, a - w + 1)
        n_days = a + 1 - s
        has_srch = S[:, s:a+1] > 0
        has_cart = CA[:, s:a+1] > 0
        has_ord  = OR[:, s:a+1] > 0
        n_triple = (has_srch & has_cart & has_ord).sum(1).astype(np.float64)
        return (n_triple / max(n_days, 1)).astype(np.float32)

    for w in [30, 90]:
        add(f"srch_cart_cooccur_{w}", lambda a, w=w: srch_cart_cooccur(a, w))
        add(f"cart_ord_cooccur_{w}", lambda a, w=w: cart_ord_cooccur(a, w))
        add(f"triple_cooccur_{w}", lambda a, w=w: triple_cooccur(a, w))

    # Trend in funnel conversion (recent vs 3 months ago)
    def ord_per_srch_trend(a):
        """Recent order/search ratio vs older: detects funnel-engagement change."""
        s0 = max(0, a - 29); s1 = max(0, a - 89)
        ord30  = OR[:, s0:a+1].sum(1).astype(np.float64)
        srch30 = S[:, s0:a+1].sum(1).astype(np.float64)
        ord90  = OR[:, s1:a+1].sum(1).astype(np.float64)
        srch90 = S[:, s1:a+1].sum(1).astype(np.float64)
        r30 = ord30 / np.maximum(srch30, 1)
        r90 = ord90 / np.maximum(srch90, 1)
        return (r30 - r90).astype(np.float32)

    add("ord_per_srch_trend_30v90", ord_per_srch_trend)

    # ── Build output ─────────────────────────────────────────────────────────
    X = {}
    for a in anchors:
        X[a] = np.stack(cols[a], axis=1).astype(np.float32)
        np.nan_to_num(X[a], copy=False, posinf=0, neginf=0)
    if verbose:
        print(f"extra_features: {len(names)} cols, anchors {anchors}", flush=True)
    return X, names
