"""two_stage v3: extended anchors + slower boosting + no calendar columns + BTYD + 9 seeds.

Confirmed on all three protocols (2026-08-13, fx_anchors_confirm): SELECT -0.0013,
VAL -0.0051, HON -0.0024 at the ensemble level vs the same-pipeline base. The VAL-sized
gain is the §12 calendar defect fixed: with months 4-8 entering training via old anchors,
eval months outside the training range (378 -> Feb, and the REAL anchor 408 -> Feb/Mar)
extrapolated into spring-level leaves; dropping tgt_*/hist_days removes the failure mode.

Config: anchors extended by 28d steps to 10 (FINAL) / 10 (PROBE), n_estimators=1200,
learning_rate=0.04, features = full store minus {tgt_month, tgt_doy_sin, tgt_doy_cos,
hist_days} plus the 5 BTYD columns (new anchors fitted deterministically via key=).

Outputs:
  submissions/two_stage_v3.csv    FINAL 9-seed log-avg at anchor 408, level EL
  model_out/t17_probe.json        probe numbers at 378 for the R* chain estimate
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
import btyd_lib as BL
from common import H, NU, N_DAYS, panel, target, rmsle, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266, 238, 210, 182, 154, 126]
TR_PROBE = [348, 320, 292, 264, 236, 208, 180, 152, 124, 96]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
SEEDS = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
DROP = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}
EL = 2.3284
SUB = Path("submissions")

LGB = dict(n_estimators=1200, learning_rate=0.04, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)

# ---- BTYD features, extending the frozen pkl deterministically where missing
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
missing = [a for a in ALL_A if a not in BT]
if missing:
    print(f"BTYD fits for {missing}")
    G = panel("gmv"); V = panel("visit")
    csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
    BUY = G > 0
    csB = np.concatenate([np.zeros((NU, 1), np.int64), BUY.astype(np.int64)], 1).cumsum(1)
    SUBS = np.random.default_rng(0).choice(NU, 25000, replace=False)
    K, KAP, DEC = 10, 0.5, 0.7

    # exact replica of t07's wG convention (matters for negative day indices at old anchors)
    wG = lambda a0, b0: csG[:, max(b0, 0) + 1] - csG[:, max(a0, 0)]

    def eb_logblocks(a):
        L = np.stack([np.log1p(wG(a - 30 * i - 29, a - 30 * i)) for i in range(K)], 1)
        w = DEC ** np.arange(K)
        pop = (L * w).sum() / (w.sum() * NU)
        return np.expm1(((L * w).sum(1) + KAP * pop) / (w.sum() + KAP))

    for a in sorted(missing):
        t0 = time.time()
        sub = V[:, :a + 1] > 0
        first = np.argmax(sub, axis=1)
        bsub = BUY[:, :a + 1]
        has = bsub.any(axis=1)
        last_buy = a - np.argmax(bsub[:, ::-1], axis=1)
        n_buy = csB[:, a + 1]
        x = np.maximum(n_buy - 1, 0).astype(np.float64)
        T = (a - first).astype(np.float64)
        t_x = np.minimum(np.where(has, last_buy - first, 0).astype(np.float64), T)
        mval = np.where(n_buy > 0, csG[:, a + 1] / np.maximum(n_buy, 1), 0.0)
        par = BL.bgnbd_fit(x[SUBS], t_x[SUBS], T[SUBS], key=a)
        gpar = BL.gg_fit(n_buy, mval, key=a)
        BT[a] = np.stack([BL.bgnbd_expected(par, x, t_x, T, H),
                          BL.bgnbd_palive(par, x, t_x, T),
                          BL.gg_expected(gpar, n_buy, mval),
                          (csB[:, a + 1] + 1.0) / (a + 1 + 20.0),
                          np.log1p(eb_logblocks(a))], 1).astype(np.float32)
        print(f"  {ds(a)} done ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(BT, open(OUT / "btyd_feats.pkl", "wb"))

X, NAMES = build_features(ALL_A)
keep = [i for i, n in enumerate(NAMES) if n not in DROP]
for a in ALL_A:
    X[a] = np.concatenate([X[a][:, keep], BT[a]], 1)
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"{X[PRED].shape[1]} features (dropped {len(DROP)} calendar cols)")


def two_stage(tr, ev, seed):
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(Y[a]) for a in tr])
    pos = lytr > 0
    clf = lgb.LGBMClassifier(**LGB, random_state=seed)
    clf.fit(Xtr, pos.astype(np.int8))
    pb = clf.predict_proba(X[ev])[:, 1]
    reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=seed)
    reg.fit(Xtr[pos], lytr[pos])
    return pb * np.clip(reg.predict(X[ev]), 0, None)


probe = []
for seed in SEEDS:
    t0 = time.time()
    lp = two_stage(TR_PROBE, FIT, seed)
    probe.append(lp)
    print(f"  probe seed {seed:>5}: {rmsle(Y[FIT], np.expm1(lp)):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
ens = np.mean(probe, 0)
p9 = rmsle(Y[FIT], np.expm1(ens))
print(f"probe ens9 {p9:.5f}   (v2 config ens9 was 1.68308)")
json.dump({"probe_ens9_v3": p9, "probe_ens9_v2": 1.68308,
           "singles": [rmsle(Y[FIT], np.expm1(v)) for v in probe]},
          open(OUT / "t17_probe.json", "w"), indent=1)

finals = []
for seed in SEEDS:
    t0 = time.time()
    finals.append(two_stage(TR_FINAL, PRED, seed))
    print(f"  final seed {seed:>5}: done ({time.time()-t0:.0f}s)", flush=True)
lf = np.mean(finals, 0)
lf = lf - lf.mean() + EL
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open(SUB / "two_stage_v3.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"two_stage_v3.csv: mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f} "
      f"p50={np.percentile(pred,50):.2f} max={pred.max():.0f}")
