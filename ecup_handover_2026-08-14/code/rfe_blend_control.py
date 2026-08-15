"""Control for rfe_blend.py: is the blend gain about feature subsets, or just seed averaging?

rfe_blend showed +0.00489 from blending seven LightGBM models trained on seven different
feature subsets. But those models share a family, hyper-parameters and training data, and
LightGBM here is stochastic (subsample 0.8, colsample 0.7). Averaging seven noisy models
reduces variance whether or not their inputs differ, so the gain has to be attributed before
it can be believed.

Three arms, all blended identically (NNLS in log space, weights on one half of the users,
scored on the other half):

  A  seven seeds, identical 165 features      <- the null: pure variance reduction
  B  seven RFE subsets, seed fixed            <- what rfe_blend measured
  C  seven RFE subsets, seeds also varied     <- both effects together

If A ≈ B the subsets add nothing and the recipe is just "average seeds", which is cheaper.
If B > A by a clear margin the feature diversity is doing real work.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
STAGES = [165, 128, 99, 77, 53, 36, 30]
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


def main():
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    X, names = build_features(anchors)
    idx = {n: i for i, n in enumerate(names)}
    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    y = target(SEL_EV)
    ly = np.log1p(y)

    meta = json.loads((OUT / "rfe.json").read_text(encoding="utf-8"))
    keep, subsets, gone = list(names), {165: list(names)}, set()
    for h in meta["history"][1:]:
        gone |= set(h["dropped"])
        if h["kept"] in STAGES:
            subsets[h["kept"]] = [n for n in names if n not in gone]

    def fit(feats, seed):
        cols = [idx[n] for n in feats]
        m = lgb.LGBMRegressor(random_state=seed, **LGB)
        m.fit(np.concatenate([X[a][:, cols] for a in SEL_TR], 0), lytr)
        return m.predict(X[SEL_EV][:, cols])

    arms = {
        "A  7 seeds, same 165 features": [(subsets[165], 42 + i) for i in range(7)],
        "B  7 RFE subsets, seed fixed":  [(subsets[k], 42) for k in STAGES],
        "C  7 RFE subsets + seeds":      [(subsets[k], 42 + i) for i, k in enumerate(STAGES)],
    }

    rng = np.random.default_rng(7)
    half = rng.random(NU) < 0.5
    ref = None
    print(f"{'arm':<32} {'best single':>12} {'blend':>10} {'gain':>10} {'mean pair corr':>15}")
    for label, spec in arms.items():
        t0 = time.time()
        P = np.stack([fit(f, s) for f, s in spec], 1)
        singles = [rmsle(y[~half], np.expm1(P[:, j][~half])) for j in range(P.shape[1])]
        w, _ = nnls(P[half], ly[half])
        s = rmsle(y[~half], np.expm1(P @ w)[~half])
        C = np.corrcoef(P.T)
        mc = float((C.sum() - len(C)) / (len(C) * (len(C) - 1)))
        if ref is None:
            ref = min(singles)
        print(f"{label:<32} {min(singles):>12.5f} {s:>10.5f} {min(singles)-s:>+10.5f} "
              f"{mc:>15.5f}   ({time.time()-t0:.0f}s)")

    print("\nread: if A's gain matches B's, the subsets are decoration and seed averaging is the"
          "\nwhole story — same result for less machinery.")


if __name__ == "__main__":
    main()
