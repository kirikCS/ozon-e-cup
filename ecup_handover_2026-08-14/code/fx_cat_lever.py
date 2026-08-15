"""Does the anchors+droptgt+slower-boosting lever transfer to CatBoost?

CatBoost is the second-heaviest measured component (1.6534) and the one seed averaging
cannot de-noise (t11). The lever's mechanisms (data volume + calendar-defect fix) are
orthogonal to seeds, so it may transfer. Single seed per cell (CatBoost is seed-stable),
SELECT + VAL.

Usage:  python -u code/fx_cat_lever.py SELECT|VAL
"""
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features

PROTO = {
    "SELECT": ([318, 290, 262, 234, 206], 348),
    "VAL": ([348, 320, 292, 264, 236], 378),
}
DROP = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}

CB = dict(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0, border_count=64,
          loss_function="RMSE", verbose=0, thread_count=23, random_seed=42,
          bootstrap_type="Bernoulli", subsample=0.8)


def extend(tr):
    out = list(tr)
    a = tr[-1] - 28
    while len(out) < 10 and a >= 60:
        out.append(a); a -= 28
    return out


def main(proto):
    TR, EV = PROTO[proto]
    TR10 = extend(TR)
    anchors = sorted(set(TR10 + [EV]))
    X, names = build_features(anchors)
    keep = np.array([i for i, n in enumerate(names) if n not in DROP])
    yev = target(EV)

    for label, tr, extra, cols in [
            ("base", TR, {}, None),
            ("ext+slow", TR10, {"iterations": 1600, "learning_rate": 0.045}, None),
            ("ext+slow+droptgt", TR10, {"iterations": 1600, "learning_rate": 0.045}, keep)]:
        t0 = time.time()
        Xtr = np.concatenate([X[a] for a in tr], 0)
        lytr = np.concatenate([np.log1p(target(a)) for a in tr])
        Xe = X[EV]
        if cols is not None:
            Xtr = Xtr[:, cols]; Xe = Xe[:, cols]
        m = CatBoostRegressor(**{**CB, **extra})
        m.fit(Xtr, lytr)
        s = rmsle(yev, np.expm1(np.clip(m.predict(Xe), 0, None)))
        print(f"  {proto} {label:<18} {s:.5f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
