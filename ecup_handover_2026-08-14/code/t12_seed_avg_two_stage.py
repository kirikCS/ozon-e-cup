"""Rebuild the two-stage+BTYD stack component with a 9-seed log-average.

Measured 2026-08-12 (fx_seed_ens, SELECT): LightGBM seed noise is real and averageable —
E[1 seed] 1.74139 -> E[9 seeds] 1.73915 for the two-stage (seeds corr 0.9967). The shipped
component (sub4 = two_stage_btyd_shift, LB R* 1.6546867) is a single seed-42 draw; its
weight in stack6 is +0.332, so de-noising it moves the stack by an estimated -0.0003.

Geometry is exactly t07's:
  FINAL  train [378, 350, 322, 294, 266] -> predict 408 (the competition anchor)
  PROBE  train [348, 320, 292, 264, 236] -> predict 378 (fully observed window)
BTYD features come from the frozen model_out/btyd_feats.pkl (all anchors cached, no fits).

The PROBE pass measures, on the shipped geometry, the seed-42 score and the 9-seed
ensemble score; their difference is the R* correction for the new component:
  R*_new = 1.6546867 - (probe_seed42 - probe_ens9)

Outputs:
  submissions/two_stage_btyd_9seed.csv   FINAL 9-seed log-averaged component
  model_out/t12_probe.json               probe numbers for the stack solve
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import H, NU, N_DAYS, target, rmsle, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266]
TR_PROBE = [348, 320, 292, 264, 236]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
SEEDS = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)

BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
missing = [a for a in ALL_A if a not in BT]
assert not missing, f"btyd_feats.pkl lacks anchors {missing}"

X, NAMES = build_features(ALL_A)
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"{X[PRED].shape[1]} features")


def two_stage(tr, ev, seed):
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(Y[a]) for a in tr])
    pos = lytr > 0
    clf = lgb.LGBMClassifier(**LGB, random_state=seed)
    clf.fit(Xtr, pos.astype(np.int8))
    pb = clf.predict_proba(X[ev])[:, 1]
    reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=seed)
    reg.fit(Xtr[pos], lytr[pos])
    return pb * np.clip(reg.predict(X[ev]), 0, None)      # log1p scale


# ---- probe: the R* correction is measured here, on the shipped geometry
probe = {}
for seed in SEEDS:
    t0 = time.time()
    lp = two_stage(TR_PROBE, FIT, seed)
    probe[seed] = lp
    print(f"  probe seed {seed:>5}: RMSLE {rmsle(Y[FIT], np.expm1(lp)):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
s42 = rmsle(Y[FIT], np.expm1(probe[42]))
ens = np.mean([probe[s] for s in SEEDS], 0)
s9 = rmsle(Y[FIT], np.expm1(ens))
delta = s42 - s9
print(f"probe seed42 {s42:.5f}  ens9 {s9:.5f}  delta {delta:+.5f}")
json.dump({"probe_seed42": s42, "probe_ens9": s9, "delta": delta,
           "probe_scores": {str(s): rmsle(Y[FIT], np.expm1(probe[s])) for s in SEEDS}},
          open(OUT / "t12_probe.json", "w"), indent=1)

# ---- final: the component itself
finals = []
for seed in SEEDS:
    t0 = time.time()
    finals.append(two_stage(TR_FINAL, PRED, seed))
    print(f"  final seed {seed:>5}: done ({time.time()-t0:.0f}s)", flush=True)
lf = np.mean(finals, 0)
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open(SUB / "two_stage_btyd_9seed.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"two_stage_btyd_9seed.csv: mean log1p={np.log1p(pred).mean():.4f} "
      f"mean={pred.mean():.2f} p50={np.percentile(pred,50):.2f} max={pred.max():.0f}")
