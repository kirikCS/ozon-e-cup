"""Does the YEAR-AGO SLICE OF THE TARGET WINDOW carry alignment signal, or is it just
old-activity (already spanned by the store)?

Only VAL can answer locally: its target window (2026-01-14..02-12) has a year-ago slice
fully inside history (days 13-42). Arms, all exact-pairing (colsample=1.0), 3 seeds:

  aligned   +3 cols from days [13, 42]   — the year-ago slice of VAL's target window
  shifted   +3 cols from days [43, 72]   — same length, one month later (control: if
                                            equal to aligned, it is generic old activity)
  noise3    width-matched noise draw

Features per window: log1p(gmv sum), buy-days, share of lifetime gmv.
If aligned > shifted beyond noise, seasonal alignment is real, and the production
component gets the aligned window of the REAL target (days 44-73), verdict via one LB slot.

Usage:  python -u code/fx_yago.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, target, rmsle, build_features, NU

TR, EV = [348, 320, 292, 264, 236], 378
SEEDS = [42, 101, 777]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")

anchors = sorted(set(TR + [EV]))
X, names = build_features(anchors)
lytr_parts = {a: np.log1p(target(a)) for a in TR}
yev = target(EV)
G = panel("gmv")
BUY = G > 0


def win_feats(s, e):
    g = G[:, s:e + 1].sum(axis=1, dtype=np.float64)
    bd = BUY[:, s:e + 1].sum(axis=1, dtype=np.float64)
    out = {}
    for a in anchors:
        tot = G[:, :a + 1].sum(axis=1, dtype=np.float64)
        out[a] = np.stack([np.log1p(g), bd, g / (tot + 1e-6)], 1).astype(np.float32)
    return out


ARMS = {
    "aligned": win_feats(13, 42),
    "shifted": win_feats(43, 72),
    "noise3": {a: np.random.default_rng(9000 + a).standard_normal((NU, 3)).astype(np.float32)
               for a in anchors},
}

base_scores = {}
for seed in SEEDS:
    t0 = time.time()
    Xtr = np.concatenate([X[a] for a in TR], 0)
    lytr = np.concatenate([lytr_parts[a] for a in TR])
    m = lgb.LGBMRegressor(**LGB, random_state=seed)
    m.fit(Xtr, lytr)
    base_scores[seed] = rmsle(yev, np.expm1(np.clip(m.predict(X[EV]), 0, None)))
    print(f"  base seed {seed:>4}: {base_scores[seed]:.5f} ({time.time()-t0:.0f}s)", flush=True)

for label, E in ARMS.items():
    deltas = []
    for seed in SEEDS:
        t0 = time.time()
        Xtr = np.concatenate([np.concatenate([X[a], E[a]], 1) for a in TR], 0)
        lytr = np.concatenate([lytr_parts[a] for a in TR])
        m = lgb.LGBMRegressor(**LGB, random_state=seed)
        m.fit(Xtr, lytr)
        s = rmsle(yev, np.expm1(np.clip(m.predict(
            np.concatenate([X[EV], E[EV]], 1)), 0, None)))
        deltas.append(s - base_scores[seed])
        print(f"  {label:<8} seed {seed:>4}: {s:.5f}  d {deltas[-1]:+.5f} ({time.time()-t0:.0f}s)",
              flush=True)
    d = np.array(deltas)
    print(f"  {label}: mean delta {d.mean():+.5f}  spread [{d.min():+.5f} .. {d.max():+.5f}]",
          flush=True)
