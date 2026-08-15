"""two_stage 9->18 seeds at FINAL geometry — clean within-pipeline chain.

The measured anchor: ts_9seed (submissions/two_stage_btyd_9seed.csv, level-matched)
with true R*_lm = sqrt(1.6556581^2 - 0.1008^2) = 1.652594 (see the level incident).
Nine more seeds shrink the seed noise; Cov(L, ens) is preserved (noise orthogonal to
the target), the variance drop is measured directly on the vectors:
    R*_18^2 = R*_9^2 - (Var(ens9) - Var(ens18))
Config identical to t12 (5 anchors, 600 trees, BTYD from pkl, level EL at the end).

Output: submissions/two_stage_btyd_18seed.csv + the implied R* printout.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import N_DAYS, NU, target, rmsle, build_features, OUT, CACHE

PRED = N_DAYS - 1
TR_FINAL = [378, 350, 322, 294, 266]
SEEDS_OLD = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
SEEDS_NEW = [11, 22, 33, 44, 55, 66, 77, 88, 99]
EL = 2.3284

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)

BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
ALL_A = sorted(set(TR_FINAL + [PRED]))
X, NAMES = build_features(ALL_A)
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
Y = {a: target(a) for a in TR_FINAL}

Xtr = np.concatenate([X[a] for a in TR_FINAL], 0)
lytr = np.concatenate([np.log1p(Y[a]) for a in TR_FINAL])
pos = lytr > 0


def two_stage(seed):
    clf = lgb.LGBMClassifier(**LGB, random_state=seed)
    clf.fit(Xtr, pos.astype(np.int8))
    pb = clf.predict_proba(X[PRED])[:, 1]
    reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=seed)
    reg.fit(Xtr[pos], lytr[pos])
    return pb * np.clip(reg.predict(X[PRED]), 0, None)


new = []
for seed in SEEDS_NEW:
    f = OUT / f"t26_final_s{seed}.npy"
    if f.exists():
        new.append(np.load(f))
        print(f"  seed {seed}: cached", flush=True)
        continue
    t0 = time.time()
    lp = two_stage(seed)
    np.save(f, lp.astype(np.float32))
    new.append(lp)
    print(f"  seed {seed}: done ({time.time()-t0:.0f}s)", flush=True)

# ens9 vector = the level-matched csv already on disk (the measured pairing's vector)
d9 = pl.read_csv("submissions/two_stage_btyd_9seed.csv")
q9 = np.log1p(d9["predict"].to_numpy())
# reconstruct raw ens9 log-scale: q9 is level-matched; the raw mean was EL - 0.1008
# ens18 = mean of [raw ens9 (recovered), 9 new seeds]; work in raw log scale then re-level
raw9 = q9 - q9.mean() + (EL - 0.1008)
ens18_raw = (raw9 * 9 + np.sum(new, 0)) / 18.0
q18 = ens18_raw - ens18_raw.mean() + EL
q9lm = q9 - q9.mean() + EL
R9 = (1.6556581195791316 ** 2 - 0.1008 ** 2) ** 0.5
R18 = float(np.sqrt(R9 ** 2 - (q9lm.var() - q18.var())))
print(f"Var ens9 {q9lm.var():.5f} -> ens18 {q18.var():.5f}")
print(f"implied R*_18 = {R18:.6f}  (R*_9 = {R9:.6f})")

pred = np.clip(np.expm1(q18), 0, None)
uids = np.load(CACHE / "uids.npy")
with open("submissions/two_stage_btyd_18seed.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"two_stage_btyd_18seed.csv written, mean log1p {np.log1p(pred).mean():.4f}")
