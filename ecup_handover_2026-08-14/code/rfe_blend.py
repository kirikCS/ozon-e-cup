"""Does feature-subset diversity buy anything in the blend?

RFE showed the single-model gain is noise (-0.0004) and that pruning barely moves the
prediction: corr to the full model stays 0.994 even after dropping 80% of the columns. But
0.994 sits inside the band the current stack components occupy (0.992-0.999), and those do
help each other — the six-vector stack beats its best single component by 0.0037. So the
question isn't settled by the correlation alone; it has to be blended and measured.

Protocol mirrors Tier 6: NNLS weights in log space fitted on one half of the users, scored on
the other half, so no number here is in-sample. Selection anchor only — VAL stays untouched.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, target, rmsle, build_features, ds, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")


def stages_from_rfe(names):
    """rebuild each RFE round's surviving feature set from the cumulative drop log"""
    meta = json.loads((OUT / "rfe.json").read_text(encoding="utf-8"))
    keep, out, gone = set(names), [], set()
    out.append((len(names), sorted(keep)))
    for h in meta["history"][1:]:
        gone |= set(h["dropped"])
        out.append((h["kept"], sorted(set(names) - gone)))
    return out


def main():
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    X, names = build_features(anchors)
    idx = {n: i for i, n in enumerate(names)}
    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    y = target(SEL_EV)
    ly = np.log1p(y)

    allst = stages_from_rfe(names)
    want = [165, 128, 99, 77, 53, 36, 30]
    stages = [(k, f) for k, f in allst if k in want]
    print(f"stages: {[k for k, _ in stages]}\n")

    P = {}
    for k, feats in stages:
        cols = [idx[n] for n in feats]
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGB)
        m.fit(np.concatenate([X[a][:, cols] for a in SEL_TR], 0), lytr)
        lp = m.predict(X[SEL_EV][:, cols])
        P[k] = lp
        print(f"  {k:>4} features  RMSLE {rmsle(y, np.expm1(lp)):.5f}  ({time.time()-t0:.0f}s)")

    keys = [k for k, _ in stages]
    M = np.stack([P[k] for k in keys], 1)
    print("\npairwise corr between stage predictions")
    print("      " + "".join(f"{k:>8}" for k in keys))
    for i, k in enumerate(keys):
        print(f"{k:>5} " + "".join(f"{np.corrcoef(M[:,i],M[:,j])[0,1]:>8.5f}" for j in range(len(keys))))

    rng = np.random.default_rng(7)
    half = rng.random(NU) < 0.5
    base = rmsle(y[~half], np.expm1(P[165][~half]))
    best_single = min(rmsle(y[~half], np.expm1(P[k][~half])) for k in keys)
    print(f"\nscored on the held-out half of users")
    print(f"  full 165 features            {base:.5f}")
    print(f"  best single stage            {best_single:.5f}")

    for label, sub in [("all 7 stages", keys), ("165 + 30", [165, 30]), ("165 + 53 + 30", [165, 53, 30]),
                       ("128 + 53 + 30", [128, 53, 30])]:
        A = np.stack([P[k] for k in sub], 1)
        w, _ = nnls(A[half], ly[half])
        s = rmsle(y[~half], np.expm1(A @ w)[~half])
        gain = base - s
        mark = "  <- beats the full model" if s < base - 1e-5 else ""
        print(f"  blend {label:<22} {s:.5f}  ({gain:+.5f}){mark}")
        print(f"        weights " + ", ".join(f"{k}:{wi:.3f}" for k, wi in zip(sub, w)))


if __name__ == "__main__":
    main()
