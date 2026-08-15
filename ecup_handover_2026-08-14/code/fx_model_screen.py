"""Model-variant screen on SELECT: single-flag LightGBM changes vs the frozen baseline.

Unlike fx_measure.py (feature groups), this varies MODEL parameters on the unchanged
165-column store. For each variant: paired seeds vs base, RMSLE delta, and corr with the
base prediction — the §6 usefulness criterion needs both (a variant that scores the same
but decorrelates is worth more than one that scores better and duplicates).

Variants from the literature sweep (2026-08-12):
  linear_tree   piecewise-LINEAR leaves extrapolate along a slope where constant leaves
                clamp — targets the §12 defect (tgt_month=2 outside the training range)
  mono_core     monotone constraints on the near-certain set only: gmv/buyday/ord window
                sums increasing, recencies decreasing; 'advanced' penalty method
  mono_broad    constraints on all window/block sums of all 8 metrics + recencies

Usage:  python -u code/fx_model_screen.py linear_tree mono_core mono_broad
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = [int(s) for s in os.environ.get("FX_SEEDS", "42,101,777").split(",")]
COLSAMPLE = float(os.environ.get("FX_COLSAMPLE", "0.7"))

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=COLSAMPLE,
           reg_lambda=5.0, n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


def mono_vector(names, broad):
    v = []
    inc_metrics = ("gmv_", "gmv_search_", "ord_", "cart_", "srch_", "visit_",
                   "engaged_", "buyday_")
    core = ("gmv_30", "gmv_90", "gmv_60", "buyday_30", "buyday_90", "buyday_all",
            "ord_30", "ord_90", "gmv_all", "ord_all")
    for n in names:
        if n.startswith("rec_"):
            v.append(-1)
        elif broad and n.startswith(inc_metrics) and ("share" not in n) and ("per" not in n):
            v.append(1)
        elif not broad and n in core:
            v.append(1)
        else:
            v.append(0)
    return v


def main(specs):
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    X, names = build_features(anchors)
    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    yev = target(SEL_EV)
    Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
    Xev = X[SEL_EV]

    VARIANTS = {
        "linear_tree": {"linear_tree": True},
        "linear_tree_l20": {"linear_tree": True, "reg_lambda": 20.0},
        "mono_core": {"monotone_constraints": mono_vector(names, False),
                      "monotone_constraints_method": "advanced"},
        "mono_broad": {"monotone_constraints": mono_vector(names, True),
                       "monotone_constraints_method": "advanced"},
        # anchor-ID feature drops (advval 2026-08-12: AUC 1.0 driven by tenure/first_day;
        # tgt_* documented in text.md §12 as clamping at the real prediction anchor).
        # Run these with FX_COLSAMPLE=1.0 — column count changes, 0.7 would re-roll draws.
        "drop_tgt": {"__drop__": ["tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"]},
        "drop_tgt_tenure": {"__drop__": ["tgt_month", "tgt_doy_sin", "tgt_doy_cos",
                                         "hist_days", "tenure", "first_day"]},
    }

    base_scores, base_preds = {}, {}
    for seed in SEEDS:
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGB, random_state=seed)
        m.fit(Xtr, lytr)
        p = m.predict(Xev)
        base_preds[seed] = p
        base_scores[seed] = rmsle(yev, np.expm1(p))
        print(f"  base seed {seed:>4}: {base_scores[seed]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    for spec in specs:
        if spec not in VARIANTS:
            print(f"  unknown variant {spec}"); continue
        scores, corrs = {}, {}
        drop = VARIANTS[spec].get("__drop__")
        cols = np.array([i for i, n in enumerate(names) if not (drop and n in drop)])
        for seed in SEEDS:
            t0 = time.time()
            params = dict(LGB)
            params.update({k: v for k, v in VARIANTS[spec].items() if k != "__drop__"})
            m = lgb.LGBMRegressor(**params, random_state=seed)
            m.fit(Xtr[:, cols], lytr)
            p = m.predict(Xev[:, cols])
            scores[seed] = rmsle(yev, np.expm1(p))
            corrs[seed] = float(np.corrcoef(p, base_preds[seed])[0, 1])
            d = scores[seed] - base_scores[seed]
            print(f"  {spec:<16} seed {seed:>4}: {scores[seed]:.5f}  delta {d:+.5f}  "
                  f"corr(base) {corrs[seed]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if seed == SEEDS[0]:
                np.save(OUT / f"fx_ms_{spec}_s{seed}.npy", p.astype(np.float32))
        deltas = np.array([scores[k] - base_scores[k] for k in SEEDS])
        print(f"  {spec}: mean delta {deltas.mean():+.5f}  "
              f"spread [{deltas.min():+.5f} .. {deltas.max():+.5f}]  "
              f"mean corr {np.mean(list(corrs.values())):.5f}", flush=True)
        row = {"ts": time.strftime("%Y-%m-%d %H:%M"), "variant": spec, "protocol": "SELECT",
               "scores": {str(k): round(v, 6) for k, v in scores.items()},
               "base": {str(k): round(v, 6) for k, v in base_scores.items()},
               "mean_delta": round(float(deltas.mean()), 6),
               "corr_base": round(float(np.mean(list(corrs.values()))), 6)}
        with open(OUT / "fx_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["linear_tree", "mono_core", "mono_broad"])
