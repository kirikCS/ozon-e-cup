"""Confirm a surviving feature group on the two held-out protocols (VAL + HON).

Run ONLY for groups that already won on SELECT — touching VAL repeatedly turns it into
a second selection anchor, which is exactly the mistake that sank the isotonic submission.

  VAL   train [348, 320, 292, 264, 236] -> evaluate 378          published base 1.69101
  HON   train [258, 230, 202, 174, 146] -> evaluate 288 (cohort) published base 1.75150

Paired seeds, same design as fx_measure.py.

Usage:
    python -u code/fx_confirm.py gap chan+funnel
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import (VAL_ANCHOR, HON_ANCHOR, target, cohort_rule, rmsle, build_features, OUT)
import fx

VAL_TR, HON_TR = [348, 320, 292, 264, 236], [258, 230, 202, 174, 146]
SEEDS = [42, 101, 777]
PUB = {"VAL": 1.69101, "HON": 1.75150}

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


def main(specs):
    anchors = sorted(set(VAL_TR + HON_TR + [VAL_ANCHOR, HON_ANCHOR]))
    t0 = time.time()
    Xb, base_names = build_features(anchors)
    print(f"base store: {len(base_names)} cols, built in {time.time()-t0:.0f}s")
    m_hon = cohort_rule(HON_ANCHOR)

    exts = {}
    for spec in specs:
        t0 = time.time()
        exts[spec] = fx.build_group(spec, anchors)
        print(f"{spec}: +{len(exts[spec][0])} cols, built in {time.time()-t0:.0f}s")

    for proto, tr, ev, mask in [("VAL", VAL_TR, VAL_ANCHOR, None),
                                ("HON", HON_TR, HON_ANCHOR, m_hon)]:
        lytr = np.concatenate([np.log1p(target(a)) for a in tr])
        yev = target(ev)
        if mask is not None:
            yev = yev[mask]

        def run(Xtr, Xev, seed):
            m = lgb.LGBMRegressor(**LGB, random_state=seed)
            m.fit(Xtr, lytr)
            p = np.expm1(m.predict(Xev))
            return rmsle(yev, p[mask] if mask is not None else p)

        Xtr_b = np.concatenate([Xb[a] for a in tr], 0)
        Xev_b = Xb[ev]
        base_scores = {}
        for seed in SEEDS:
            t0 = time.time()
            s = run(Xtr_b, Xev_b, seed)
            base_scores[seed] = s
            flag = f"   published {PUB[proto]}  delta {s-PUB[proto]:+.5f}" if seed == 42 else ""
            print(f"  {proto} base seed {seed:>4}: {s:.5f}  ({time.time()-t0:.0f}s){flag}")

        for spec in specs:
            ext_names, Xe = exts[spec]
            Xtr = np.concatenate([np.concatenate([Xb[a], Xe[a]], 1) for a in tr], 0)
            Xev = np.concatenate([Xb[ev], Xe[ev]], 1)
            scores = {}
            for seed in SEEDS:
                t0 = time.time()
                s = run(Xtr, Xev, seed)
                scores[seed] = s
                print(f"  {proto} {spec:<14} seed {seed:>4}: {s:.5f}  "
                      f"paired delta {s-base_scores[seed]:+.5f}  ({time.time()-t0:.0f}s)")
            deltas = np.array([scores[k] - base_scores[k] for k in SEEDS])
            print(f"  {proto} {spec}: mean delta {deltas.mean():+.5f}  "
                  f"spread [{deltas.min():+.5f} .. {deltas.max():+.5f}]")
            row = {"ts": time.strftime("%Y-%m-%d %H:%M"), "spec": spec, "protocol": proto,
                   "scores": {str(k): round(v, 6) for k, v in scores.items()},
                   "base": {str(k): round(v, 6) for k, v in base_scores.items()},
                   "mean_delta": round(float(deltas.mean()), 6)}
            with open(OUT / "fx_results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            del Xtr, Xev


if __name__ == "__main__":
    main(sys.argv[1:])
