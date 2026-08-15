"""Is the blend gain ensembling, or just a slope fit wearing an ensemble costume?

rfe_blend_control found that seven seeds of the same model blend to +0.0053, slightly beating
seven different feature subsets. Before calling that "averaging works", note what NNLS does to
seven columns correlated at 0.996: it is very close to fitting a single scalar on log1p(p).
That is slope calibration, and slope calibration is known to be worth a lot here on its own —
expm1(0.55*log1p(prev30) + 0.9) takes the naive baseline from 2.195 to 1.901 with no features
at all.

So decompose the +0.0053 into the part any single model gets for free from a slope, and the
part that genuinely comes from averaging. Everything is fitted on one half of the users and
scored on the other, so the comparison is fair across arms.
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, target, rmsle, build_features

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")
SEEDS = 7


def main():
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    X, _ = build_features(anchors)
    Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    y = target(SEL_EV)
    ly = np.log1p(y)

    P = []
    for s in range(SEEDS):
        t0 = time.time()
        m = lgb.LGBMRegressor(random_state=42 + s, **LGB)
        m.fit(Xtr, lytr)
        P.append(m.predict(X[SEL_EV]))
        print(f"  seed {42+s} fitted ({time.time()-t0:.0f}s)")
    P = np.stack(P, 1)

    rng = np.random.default_rng(7)
    half = rng.random(NU) < 0.5
    tr, ev = half, ~half

    def score(v):
        return rmsle(y[ev], np.expm1(v[ev]))

    p1 = P[:, 0]
    mean7 = P.mean(1)

    rows = []
    rows.append(("одна модель, как есть", score(p1)))

    b = float(nnls(p1[tr, None], ly[tr])[0][0])
    rows.append((f"одна модель + наклон (b={b:.4f})", score(b * p1)))

    A = np.stack([p1, np.ones(NU)], 1)
    w2, _ = nnls(A[tr], ly[tr])
    rows.append((f"одна модель + наклон и сдвиг (b={w2[0]:.4f}, c={w2[1]:.4f})", score(A @ w2)))

    rows.append(("среднее 7 сидов, как есть", score(mean7)))

    bm = float(nnls(mean7[tr, None], ly[tr])[0][0])
    rows.append((f"среднее 7 сидов + наклон (b={bm:.4f})", score(bm * mean7)))

    w7, _ = nnls(P[tr], ly[tr])
    rows.append((f"NNLS по 7 сидам (сумма весов {w7.sum():.4f})", score(P @ w7)))

    base = rows[0][1]
    print(f"\n{'вариант':<52}{'RMSLE':>10}{'к одной модели':>16}")
    for label, s in rows:
        print(f"{label:<52}{s:>10.5f}{base-s:>+16.5f}")

    slope_only = base - rows[1][1]
    full = base - rows[-1][1]
    print(f"\nиз общего выигрыша {full:+.5f} наклон объясняет {slope_only:+.5f} "
          f"({100*slope_only/full:.0f}%), усреднение — остальные {full-slope_only:+.5f}")


if __name__ == "__main__":
    main()
