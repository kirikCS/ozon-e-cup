"""Measure a feature-extension group on the SELECT protocol with paired seeds.

Protocol (decisions are made here and only here):
  SELECT  train [318, 290, 262, 234, 206] -> evaluate 348
Published full-store baseline at seed 42: RMSLE 1.74216 (feat_select.py, rfe.py).
The base battery must reproduce that number exactly — if it does not, the harness is
broken and nothing measured with it counts.

Pairing caveat (verified adversarially 2026-08-12): at colsample_bytree=0.7 adding
columns re-rolls every per-tree column draw, so the "paired" delta has ~zero covariance
between arms and pairing cancels nothing — sd(delta) = sqrt(2)*sd(single seed) ≈ 0.0006,
and a 3-seed mean cannot resolve effects below ~0.001. Run the screen with FX_COLSAMPLE=1.0
instead: without column sampling both arms grow identical trees until an ext column first
wins a split, and the delta becomes the direct feature effect. The noiseK@D groups
(K columns, draw D) are the spurious-split control and must be width-matched to the
group under test.

Usage:
    python -u code/fx_measure.py base gap dow chan funnel noise16 chan+funnel
Seeds are fixed: 42, 101, 777.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, ds, OUT
import fx

import os
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = [int(s) for s in os.environ.get("FX_SEEDS", "42,101,777").split(",")]
BASE_PUBLISHED = 1.74216

# FX_COLSAMPLE=1.0 turns off column sampling: base and base+ext trees are then identical
# until the first split that actually uses an ext column, so the paired delta measures the
# feature effect directly instead of a re-rolled colsample draw (review finding: at 0.7 the
# paired delta has covariance ~0 between arms and pairing cancels nothing). The published
# 1.74216 check only applies at the production value 0.7.
COLSAMPLE = float(os.environ.get("FX_COLSAMPLE", "0.7"))

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=COLSAMPLE,
           reg_lambda=5.0, n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


# FX_MODEL=2s screens with the exact two-stage decomposition (t07's f_two_stage):
# P(y>0) classifier x E[log1p y | y>0] regressor. The classifier is a different learning
# problem, so features useless for direct L2 may still move it.
MODEL = os.environ.get("FX_MODEL", "l2")


def fit_eval(Xtr, lytr, Xev, yev, seed):
    if MODEL == "2s":
        pos = lytr > 0
        cp = {k: v for k, v in LGB.items() if k != "objective"}
        clf = lgb.LGBMClassifier(**cp, random_state=seed)
        clf.fit(Xtr, pos.astype(np.int8))
        pb = clf.predict_proba(Xev)[:, 1]
        reg = lgb.LGBMRegressor(**LGB, random_state=seed)
        reg.fit(Xtr[pos], lytr[pos])
        p = np.expm1(pb * np.clip(reg.predict(Xev), 0, None))
        return rmsle(yev, p), reg
    m = lgb.LGBMRegressor(**LGB, random_state=seed)
    m.fit(Xtr, lytr)
    return rmsle(yev, np.expm1(m.predict(Xev))), m


def main(specs):
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    t0 = time.time()
    Xb, base_names = build_features(anchors)
    print(f"base store: {len(base_names)} cols, built in {time.time()-t0:.0f}s")

    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    yev = target(SEL_EV)
    Xtr_base = np.concatenate([Xb[a] for a in SEL_TR], 0)
    Xev_base = Xb[SEL_EV]

    results = {}
    base_scores = {}
    for seed in SEEDS:
        t0 = time.time()
        s, _ = fit_eval(Xtr_base, lytr, Xev_base, yev, seed)
        base_scores[seed] = s
        flag = ""
        if seed == 42 and COLSAMPLE == 0.7:
            flag = f"   published {BASE_PUBLISHED}  delta {s-BASE_PUBLISHED:+.5f}"
            if abs(s - BASE_PUBLISHED) > 2e-5:
                flag += "   <<< HARNESS BROKEN"
        print(f"  base seed {seed:>4}: {s:.5f}  ({time.time()-t0:.0f}s){flag}")
    results["base"] = base_scores

    for spec in specs:
        if spec == "base":
            continue
        t0 = time.time()
        ext_names, Xe = fx.build_group(spec, anchors)
        tb = time.time() - t0
        Xtr = np.concatenate([np.concatenate([Xb[a], Xe[a]], 1) for a in SEL_TR], 0)
        Xev = np.concatenate([Xb[SEL_EV], Xe[SEL_EV]], 1)
        scores, gains = {}, None
        for seed in SEEDS:
            t0 = time.time()
            s, m = fit_eval(Xtr, lytr, Xev, yev, seed)
            scores[seed] = s
            if seed == 42:
                imp = m.booster_.feature_importance("gain")
                tot = imp.sum()
                gains = {n: float(imp[len(base_names) + i] / tot)
                         for i, n in enumerate(ext_names)}
            d = s - base_scores[seed]
            print(f"  {spec:<14} seed {seed:>4}: {s:.5f}  paired delta {d:+.5f}  ({time.time()-t0:.0f}s)")
        deltas = np.array([scores[k] - base_scores[k] for k in SEEDS])
        top = sorted(gains.items(), key=lambda kv: -kv[1])[:5] if gains else []
        print(f"  {spec}: +{len(ext_names)} cols (built {tb:.0f}s) | "
              f"mean delta {deltas.mean():+.5f}  spread [{deltas.min():+.5f} .. {deltas.max():+.5f}]")
        if top:
            print(f"    top gain-share: " + ", ".join(f"{n} {100*v:.2f}%" for n, v in top))
        results[spec] = scores
        row = {"ts": time.strftime("%Y-%m-%d %H:%M"), "spec": spec, "protocol": "SELECT",
               "colsample": COLSAMPLE, "model": MODEL,
               "n_ext": len(ext_names), "scores": {str(k): round(v, 6) for k, v in scores.items()},
               "base": {str(k): round(v, 6) for k, v in base_scores.items()},
               "mean_delta": round(float(deltas.mean()), 6),
               "top_gain": {n: round(v, 5) for n, v in top}}
        with open(OUT / "fx_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        del Xtr, Xev, Xe

    print("\nsummary (mean paired delta vs base, negative = better):")
    for spec, sc in results.items():
        if spec == "base":
            continue
        d = np.mean([sc[k] - base_scores[k] for k in SEEDS])
        print(f"  {spec:<16} {d:+.5f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["base"])
