"""Feature pruning: how sparse can the tree model get before it starts losing?

The selection anchor is deliberately NOT the validation anchor. Choosing features by the
score at anchor 378 and then quoting that same score is the mistake that made the isotonic
submission look best on validation and land worst on the leaderboard. So:

  SELECT  train [318, 290, 262, 234, 206] -> evaluate 348   <- every decision is made here
  VAL     train [348, 320, 292, 264, 236] -> evaluate 378   <- touched once, at the end
  HON     train [258, 230, 202, 174, 146] -> evaluate 288   <- touched once, at the end

Anchor spacing is 28 days and the evaluation anchor sits 30 days after the last training
anchor, so no training target window overlaps the evaluation one — same geometry as Tier 3.

Usage:
    python code/feat_select.py sweep     # rank by gain, drop the tail, plot the curve
    python code/feat_select.py confirm K # re-measure the best K on VAL and HON
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import (VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, build_features,
                    ds, OUT)

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
VAL_TR, VAL_EV = [348, 320, 292, 264, 236], VAL_ANCHOR
HON_TR, HON_EV = [258, 230, 202, 174, 146], HON_ANCHOR

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")


def fit_eval(X, names, tr, ev, keep, mask=None, want_importance=False):
    """keep = boolean mask over columns; returns (rmsle, gain importance or None)"""
    cols = np.flatnonzero(keep)
    Xtr = np.concatenate([X[a][:, cols] for a in tr], 0)
    lytr = np.concatenate([np.log1p(target(a)) for a in tr])
    m = lgb.LGBMRegressor(**LGB)
    m.fit(Xtr, lytr)
    p = np.expm1(m.predict(X[ev][:, cols]))
    y = target(ev)
    s = rmsle(y, p) if mask is None else rmsle(y[mask], p[mask])
    imp = None
    if want_importance:
        imp = dict(zip([names[i] for i in cols], m.booster_.feature_importance("gain")))
    return s, imp


def sweep():
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    t0 = time.time()
    X, names = build_features(anchors)
    print(f"feature store: {len(names)} columns, built in {time.time()-t0:.0f}s\n")

    keep = np.ones(len(names), bool)
    t0 = time.time()
    base, imp = fit_eval(X, names, SEL_TR, SEL_EV, keep, want_importance=True)
    print(f"baseline at {ds(SEL_EV)}: {len(names)} features -> RMSLE {base:.5f}  ({time.time()-t0:.0f}s)\n")

    order = sorted(names, key=lambda n: imp.get(n, 0.0))        # least useful first
    dead = [n for n in order if imp.get(n, 0.0) == 0.0]
    print(f"{len(dead)} features carry zero gain and can go for free\n")

    rows = [{"n_features": len(names), "dropped": 0, "rmsle": base}]
    print(f"{'kept':>6} {'dropped':>8} {'RMSLE':>9} {'vs base':>9}")
    print(f"{len(names):>6} {0:>8} {base:>9.5f} {0.0:>+9.5f}   <- baseline")
    for k in [10, 20, 30, 45, 60, 75, 90, 105, 120, 135, 145]:
        if k >= len(names):
            break
        drop = set(order[:k])
        keep = np.array([n not in drop for n in names])
        s, _ = fit_eval(X, names, SEL_TR, SEL_EV, keep)
        rows.append({"n_features": int(keep.sum()), "dropped": k, "rmsle": s})
        flag = "  <- better" if s < base else ""
        print(f"{keep.sum():>6} {k:>8} {s:>9.5f} {s-base:>+9.5f}{flag}")

    best = min(rows, key=lambda r: r["rmsle"])
    out = {"selection_anchor": ds(SEL_EV), "baseline": base, "rows": rows,
           "ranking_worst_first": order, "zero_gain": dead, "best": best}
    (OUT / "feat_select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nbest: {best['n_features']} features, RMSLE {best['rmsle']:.5f} "
          f"({best['rmsle']-base:+.5f} vs the full set)")
    print(f"ranking and curve written to {OUT/'feat_select.json'}")


def confirm(k):
    """re-measure a pruned set on the two protocols that selection never touched"""
    meta = json.loads((OUT / "feat_select.json").read_text())
    order = meta["ranking_worst_first"]
    anchors = sorted(set(VAL_TR + HON_TR + [VAL_EV, HON_EV]))
    X, names = build_features(anchors)
    drop = set(order[:k])
    keep = np.array([n not in drop for n in names])
    print(f"\nconfirming {int(keep.sum())} features (dropped {k}) on the held-out protocols")
    print("published full-feature reference: VAL 1.69101, HON 1.75150\n")
    m_hon = cohort_rule(HON_EV)
    for label, tr, ev, mask, ref in [("VAL", VAL_TR, VAL_EV, None, 1.69101),
                                     ("HON", HON_TR, HON_EV, m_hon, 1.75150)]:
        s, _ = fit_eval(X, names, tr, ev, keep, mask=mask)
        print(f"  {label}  anchor {ds(ev)}  RMSLE {s:.5f}   full set {ref}   delta {s-ref:+.5f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        sweep()
    elif cmd == "confirm":
        confirm(int(sys.argv[2]))
    else:
        print(__doc__)
