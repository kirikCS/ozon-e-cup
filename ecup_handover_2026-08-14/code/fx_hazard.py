"""Discrete-time daily buy-hazard as a feature source (digest idea #4, first cut).

Person-day model: P(buy | user-state at day d, calendar of d), LightGBM logloss on a
50k-user sample. Leak discipline: the hazard model trains ONLY on days <= 206 (before the
earliest SELECT training anchor), so no downstream anchor's target window is ever seen.
Downstream features per anchor a: over the 30 target days, with user state frozen at a:
    haz_ebuy  = log1p(sum_d h(u, d))          (expected buy-days)
    haz_lp0   = sum_d log(1 - h(u, d))        (log P(no purchase))
Screened with exact pairing on SELECT vs a width-matched noise control.

This first cut cannot see holiday response (no NY inside days <= 206) — it answers only
"does a hazard-shaped intensity feature add anything at all". Kill: delta within noise.

Usage:  python -u code/fx_hazard.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, target, rmsle, build_features, NU, D0

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = [42, 101, 777]
HZ_DAYS = (40, 206)          # hazard training days
N_USERS = 50_000

G = panel("gmv")
V = panel("visit")
BUY = (G > 0).astype(np.int8)

# --- per-(user, day) state, vectorised over the full panel
T = 409
day_idx = np.arange(T)
# last buy day before d (exclusive): ffill of buy-day indices, shifted by one day
lastbuy = np.where(BUY > 0, day_idx[None, :], -1)
lastbuy = np.maximum.accumulate(lastbuy, axis=1)
csB = np.concatenate([np.zeros((NU, 1), np.int32), BUY.astype(np.int32)], 1).cumsum(1)
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
csV = np.concatenate([np.zeros((NU, 1), np.int32), V.astype(np.int32)], 1).cumsum(1)
first_visit = np.where(V.any(1), np.argmax(V > 0, axis=1), T)

WD0 = D0.weekday()
dow = (day_idx + WD0) % 7
doy = (day_idx + 0) % 365
HOLI = np.zeros(T, np.float32)
for s, e in [(48, 58), (61, 71), (309, 318), (348, 371)]:
    HOLI[s:min(e + 1, T)] = 1.0


def state_feats(users, d_state, d_cal):
    """features for users with state frozen at d_state, calendar of day d_cal"""
    lb = lastbuy[users, d_state]
    rec = np.where(lb >= 0, d_state - lb, 999).astype(np.float32)
    nb = csB[users, d_state + 1].astype(np.float32)
    g30 = (csG[users, d_state + 1] - csG[users, max(0, d_state - 29)]).astype(np.float32)
    v30 = (csV[users, d_state + 1] - csV[users, max(0, d_state - 29)]).astype(np.float32)
    ten = np.maximum(d_state - first_visit[users], 0).astype(np.float32)
    cal = np.array([np.sin(2 * np.pi * doy[d_cal] / 365.25),
                    np.cos(2 * np.pi * doy[d_cal] / 365.25),
                    float(dow[d_cal] >= 5), HOLI[d_cal]], np.float32)
    return np.column_stack([np.log1p(rec), np.log1p(nb), np.log1p(g30), np.log1p(v30),
                            np.log1p(ten),
                            np.repeat(cal[None, :], len(users), 0)])


rng = np.random.default_rng(0)
sample = rng.choice(NU, N_USERS, replace=False)

# training table: every 2nd day per user in HZ_DAYS (thin for speed)
t0 = time.time()
rows_X, rows_y = [], []
for d in range(HZ_DAYS[0], HZ_DAYS[1] + 1, 2):
    rows_X.append(state_feats(sample, d - 1, d))     # state up to yesterday, predict today
    rows_y.append(BUY[sample, d])
Xh = np.concatenate(rows_X, 0)
yh = np.concatenate(rows_y, 0)
print(f"hazard table {Xh.shape}, positive rate {yh.mean():.4f}, built {time.time()-t0:.0f}s",
      flush=True)

t0 = time.time()
hz = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63,
                        min_child_samples=200, max_bin=63, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, reg_lambda=5.0, n_jobs=23, verbose=-1,
                        random_state=42, force_col_wise=True)
hz.fit(Xh, yh)
print(f"hazard model fitted ({time.time()-t0:.0f}s)", flush=True)

anchors = sorted(set(SEL_TR + [SEL_EV]))


def hazard_feats(a):
    ebuy = np.zeros(NU)
    lp0 = np.zeros(NU)
    allu = np.arange(NU)
    for d in range(a + 1, a + 31):
        h = hz.predict_proba(state_feats(allu, a, d))[:, 1]
        ebuy += h
        lp0 += np.log(np.clip(1 - h, 1e-9, 1))
    return np.stack([np.log1p(ebuy), lp0], 1).astype(np.float32)


t0 = time.time()
HF = {a: hazard_feats(a) for a in anchors}
print(f"hazard features for {len(anchors)} anchors ({time.time()-t0:.0f}s)", flush=True)

X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")

NOISE = {a: np.random.default_rng(9000 + a).standard_normal((NU, 2)).astype(np.float32)
         for a in anchors}
base_scores = {}
for arm, E in [("base", None), ("noise2", NOISE), ("hazard", HF)]:
    deltas = []
    for seed in SEEDS:
        t0 = time.time()
        if E is None:
            Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
            Xev = X[SEL_EV]
        else:
            Xtr = np.concatenate([np.concatenate([X[a], E[a]], 1) for a in SEL_TR], 0)
            Xev = np.concatenate([X[SEL_EV], E[SEL_EV]], 1)
        m = lgb.LGBMRegressor(**LGB, random_state=seed)
        m.fit(Xtr, lytr)
        s = rmsle(yev, np.expm1(np.clip(m.predict(Xev), 0, None)))
        if E is None:
            base_scores[seed] = s
            print(f"  base   seed {seed:>4}: {s:.5f} ({time.time()-t0:.0f}s)", flush=True)
        else:
            deltas.append(s - base_scores[seed])
            print(f"  {arm:<6} seed {seed:>4}: {s:.5f}  d {deltas[-1]:+.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if E is not None:
        d = np.array(deltas)
        print(f"  {arm}: mean delta {d.mean():+.5f}  spread [{d.min():+.5f} .. {d.max():+.5f}]",
              flush=True)
