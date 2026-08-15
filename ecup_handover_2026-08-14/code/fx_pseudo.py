"""Pseudo-labelling the evaluation anchor: does training on soft-labelled rows at the
prediction distribution fix common-mode extrapolation? (Literature sweep item #3.)

SELECT replica: student trains on [318, 290, 262, 234, 206] plus the 250k rows at anchor
348 labelled by a teacher, weight w; evaluated on the true target at 348.
For the real submission the teacher at anchor 408 would be stack6 itself.

Arms (each vs the 3-seed base):
  t9_w03 / t9_w05   teacher = 9-seed L2 ensemble at 348 (best local vector), w=0.3 / 0.5
  self_w05          teacher = the student's own single-seed prediction — separates
                    "rows at the eval distribution" from "ensemble knowledge transfer"
  t9tr_w05          teacher rows at TRAINING anchor 318 instead of 348 — separates
                    "eval-distribution placement" from generic soft-label augmentation

Kill criteria: gain <= seed spread; or self_w05 reproduces the full effect (row-weighting,
not teacher signal); or t9tr_w05 == t9_w05 (not about the eval distribution).
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = [42, 101, 777]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
Xev = X[SEL_EV]

teacher9 = np.mean([np.load(OUT / f"fx_se_l2_s{s}.npy")
                    for s in [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]], 0)

base_scores, base_pred = {}, {}
for seed in SEEDS:
    t0 = time.time()
    m = lgb.LGBMRegressor(**LGB, random_state=seed)
    m.fit(Xtr, lytr)
    p = m.predict(Xev)
    base_pred[seed] = p
    base_scores[seed] = rmsle(yev, np.expm1(p))
    print(f"  base seed {seed:>4}: {base_scores[seed]:.5f}  ({time.time()-t0:.0f}s)", flush=True)


lyev = np.log1p(yev)


def lvl(p):
    """RMSLE after the optimal level shift — production sets the level exactly (EL from the
    LB parabola), so any gain that vanishes here is worthless for the stack."""
    c = lyev.mean() - p.mean()
    return rmsle(yev, np.expm1(np.clip(p + c, 0, None)))


def slope(p):
    """RMSLE after optimal slope+level (b, c) — the blend_confound control."""
    A = np.stack([p, np.ones_like(p)], 1)
    b, c = np.linalg.lstsq(A, lyev, rcond=None)[0]
    return rmsle(yev, np.expm1(np.clip(b * p + c, 0, None)))


def arm(label, pseudo_X, pseudo_y_fn, w):
    scores = {}
    for seed in SEEDS:
        t0 = time.time()
        py = pseudo_y_fn(seed)
        Xa = np.concatenate([Xtr, pseudo_X], 0)
        ya = np.concatenate([lytr, py])
        sw = np.concatenate([np.ones(len(lytr)), np.full(len(py), w)])
        m = lgb.LGBMRegressor(**LGB, random_state=seed)
        m.fit(Xa, ya, sample_weight=sw)
        p = m.predict(Xev)
        s = rmsle(yev, np.expm1(p))
        scores[seed] = s
        pb = base_pred[seed]
        print(f"  {label:<10} seed {seed:>4}: {s:.5f} (d {s-base_scores[seed]:+.5f}) | "
              f"lvl {lvl(p):.5f} vs base {lvl(pb):.5f} (d {lvl(p)-lvl(pb):+.5f}) | "
              f"slope {slope(p):.5f} vs {slope(pb):.5f} (d {slope(p)-slope(pb):+.5f})  "
              f"({time.time()-t0:.0f}s)", flush=True)
    d = np.array([scores[k] - base_scores[k] for k in SEEDS])
    print(f"  {label}: mean raw delta {d.mean():+.5f}  spread [{d.min():+.5f} .. {d.max():+.5f}]",
          flush=True)


arm("t9_w03", Xev, lambda seed: teacher9, 0.3)
arm("t9_w05", Xev, lambda seed: teacher9, 0.5)
arm("self_w05", Xev, lambda seed: base_pred[seed], 0.5)
# (the placement control — teacher rows at a training anchor — needs teacher predictions
#  at 318, i.e. another 9-seed battery; run it only if the arms above show an effect)
print("done")
