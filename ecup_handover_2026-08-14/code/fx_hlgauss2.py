"""Does the binned-classification model add signal BEYOND the 9-seed ensembles?

fx_hlgauss showed: parity with L2 standalone, corr 0.9948 (below the seed floor 0.996),
half/half blend -0.0018 beyond slope. But a single extra L2 seed gives a similar-size
blend gain, so the decisive contrast is against the already-averaged vectors:

  half/half OLS combos (weights on half A, scored on half B, 3 random splits):
    ens9_l2 + slope                  <- control: what slope alone is worth
    ens9_l2 + extra L2 seed (9090)   <- control: what one more seed-alike is worth
    ens9_l2 + ens9_2s                <- known diversity: second architecture
    ens9_l2 + hlg_ens3               <- the candidate
    ens9_l2 + ens9_2s + hlg_ens3     <- does it stack on top of both?

Usage:  python -u code/fx_hlgauss2.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
NBINS = 30
HL_SEEDS = [42, 101, 777]

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
yev = target(SEL_EV)
ly = np.log1p(yev)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
Xev = X[SEL_EV]

pos = lytr > 0
qs = np.quantile(lytr[pos], np.linspace(0, 1, NBINS)[1:-1])
lab = np.where(pos, 1 + np.searchsorted(qs, lytr), 0).astype(np.int32)
centers = np.array([lytr[lab == k].mean() if (lab == k).any() else 0.0
                    for k in range(NBINS)])

hl = []
for s in HL_SEEDS:
    f = OUT / f"fx_hlg_s{s}.npy"
    if f.exists():
        hl.append(np.load(f))
        continue
    t0 = time.time()
    m = lgb.LGBMClassifier(objective="multiclass", num_class=NBINS, n_estimators=200,
                           learning_rate=0.15, num_leaves=63, min_child_samples=100,
                           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=5.0, n_jobs=23, verbose=-1, random_state=s,
                           force_col_wise=True)
    m.fit(Xtr, lab)
    lp = m.predict_proba(Xev) @ centers
    np.save(f, lp.astype(np.float32))
    hl.append(lp)
    print(f"  hlg seed {s}: {rmsle(yev, np.expm1(np.clip(lp,0,None))):.5f} "
          f"({time.time()-t0:.0f}s)", flush=True)
hlg = np.mean(hl, 0)

l2_seeds = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
ens_l2 = np.mean([np.load(OUT / f"fx_se_l2_s{s}.npy") for s in l2_seeds], 0)
ens_2s = np.mean([np.load(OUT / f"fx_se_2s_s{s}.npy") for s in l2_seeds], 0)
extra_seed = np.load(OUT / "fx_se_l2_s9090.npy")

print(f"standalone: ens9_l2 {rmsle(yev, np.expm1(np.clip(ens_l2,0,None))):.5f}  "
      f"ens9_2s {rmsle(yev, np.expm1(np.clip(ens_2s,0,None))):.5f}  "
      f"hlg_ens3 {rmsle(yev, np.expm1(np.clip(hlg,0,None))):.5f}")
print(f"corr(hlg, ens9_l2) {np.corrcoef(hlg, ens_l2)[0,1]:.5f}   "
      f"corr(ens9_2s, ens9_l2) {np.corrcoef(ens_2s, ens_l2)[0,1]:.5f}")

COMBOS = [
    ("ens9_l2 + slope", [ens_l2]),
    ("ens9_l2 + seed9090", [ens_l2, extra_seed]),
    ("ens9_l2 + ens9_2s", [ens_l2, ens_2s]),
    ("ens9_l2 + hlg", [ens_l2, hlg]),
    ("l2 + 2s + hlg", [ens_l2, ens_2s, hlg]),
]
for label, vecs in COMBOS:
    scores = []
    for r in range(3):
        half = np.random.default_rng(r).random(len(ly)) < 0.5
        A = np.stack(vecs + [np.ones_like(ly)], 1)
        w = np.linalg.lstsq(A[half], ly[half], rcond=None)[0]
        p = A[~half] @ w
        scores.append(rmsle(yev[~half], np.expm1(np.clip(p, 0, None))))
    print(f"  {label:<22} held-out {np.mean(scores):.5f} +- {np.std(scores):.5f}")
