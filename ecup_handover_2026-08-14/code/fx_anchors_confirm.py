"""Confirm the anchor-count + trees/LR lever on VAL and HON (L2 and two-stage).

Extended anchor sets step back by 28d from the base five while history stays >= 60 days
(HON's older anchors run out early — the set is capped there naturally).
Compared at the 3-seed-ensemble level, which is what the production chain uses.

Usage:  python -u code/fx_anchors_confirm.py VAL
        python -u code/fx_anchors_confirm.py HON
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, cohort_rule, OUT

PROTO = {
    "SELECT": ([318, 290, 262, 234, 206], 348, False),
    "VAL": ([348, 320, 292, 264, 236], 378, False),
    "HON": ([258, 230, 202, 174, 146], 288, True),
}
SEEDS = [42, 101, 777]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)


def extend(tr, n_extra):
    out = list(tr)
    a = tr[-1] - 28
    while len(out) < len(tr) + n_extra and a >= 60:
        out.append(a)
        a -= 28
    return out


def main(proto):
    TR, EV, use_mask = PROTO[proto]
    TR10 = extend(TR, 5)
    anchors = sorted(set(TR10 + [EV]))
    X, _ = build_features(anchors)
    yev = target(EV)
    mask = cohort_rule(EV) if use_mask else None
    yv = yev[mask] if mask is not None else yev
    Xev = X[EV]

    def score(v):
        pv = v[mask] if mask is not None else v
        return rmsle(yv, np.expm1(np.clip(pv, 0, None)))

    print(f"{proto}: base anchors {TR}, extended {TR10}")

    # VAL exploded with extended anchors (2s ens 1.73184 vs 1.68699): the §12 calendar
    # defect — added anchors bring months 4-8 into training, eval month 2 extrapolates
    # into a spring-level leaf. The droptgt variant removes the calendar/anchor-ID
    # columns entirely (SELECT price: +0.0003 noise).
    import os
    drop_names = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}
    _, names_all = build_features([EV])  # names only; cached features reused
    keep_cols = np.array([i for i, n in enumerate(names_all) if n not in drop_names])

    for model in ["l2", "2s"]:
        for label, tr, extra, cols in [
                ("base", TR, {}, None),
                ("ext+n1200", TR10, {"n_estimators": 1200, "learning_rate": 0.04}, None),
                ("ext+n1200+droptgt", TR10,
                 {"n_estimators": 1200, "learning_rate": 0.04}, keep_cols)]:
            Xtr = np.concatenate([X[a] for a in tr], 0)
            lytr = np.concatenate([np.log1p(target(a)) for a in tr])
            Xe = Xev
            if cols is not None:
                Xtr = Xtr[:, cols]
                Xe = Xev[:, cols]
            preds = []
            for seed in SEEDS:
                t0 = time.time()
                params = dict(LGB); params.update(extra)
                if model == "2s":
                    pos = lytr > 0
                    clf = lgb.LGBMClassifier(**params, random_state=seed)
                    clf.fit(Xtr, pos.astype(np.int8))
                    pb = clf.predict_proba(Xe)[:, 1]
                    reg = lgb.LGBMRegressor(**params, objective="regression",
                                            random_state=seed)
                    reg.fit(Xtr[pos], lytr[pos])
                    lp = pb * np.clip(reg.predict(Xe), 0, None)
                else:
                    m = lgb.LGBMRegressor(**params, objective="regression",
                                          random_state=seed)
                    m.fit(Xtr, lytr)
                    lp = m.predict(Xe)
                preds.append(lp)
                print(f"  {model} {label:<10} seed {seed:>4}: {score(lp):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            ens = score(np.mean(preds, 0))
            print(f"  {model} {label}: 3-seed ens {ens:.5f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
