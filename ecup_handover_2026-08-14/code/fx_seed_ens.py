"""Pure seed-ensembling curve for the tabular components: does an equal-weight log-average
of K seeds beat one seed, without the NNLS slope confound?

Background: rfe_blend_control branch A showed a 7-seed NNLS blend at 1.73860 vs best single
1.74385, and blend_confound attributed 44% (~-0.0023) to variance averaging. But stack6's
tabular components are all single-seed, and t11 closed seed averaging only for CatBoost
(seeds corr 0.998 there). This measures the clean curve: subset-averaged E[ens_k] like the
CNN seed curve, for the plain L2 LGBM and the two-stage model.

Usage:  python -u code/fx_seed_ens.py l2            # SELECT (decision protocol)
        python -u code/fx_seed_ens.py 2s VAL        # confirmation only
        python -u code/fx_seed_ens.py 2s HON
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, cohort_rule, OUT

PROTO = {
    "SELECT": ([318, 290, 262, 234, 206], 348, False),
    "VAL": ([348, 320, 292, 264, 236], 378, False),
    "HON": ([258, 230, 202, 174, 146], 288, True),
}
SEEDS = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


def main(model, proto="SELECT"):
    TR, EV, use_mask = PROTO[proto]
    anchors = sorted(set(TR + [EV]))
    X, names = build_features(anchors)
    lytr = np.concatenate([np.log1p(target(a)) for a in TR])
    yev = target(EV)
    mask = cohort_rule(EV) if use_mask else None
    if mask is not None:
        yev = yev[mask]
    Xtr = np.concatenate([X[a] for a in TR], 0)
    Xev = X[EV]

    preds = []          # log1p-scale predictions per seed
    for seed in SEEDS:
        f = OUT / f"fx_se_{model}_{proto}_s{seed}.npy"
        if f.exists():
            preds.append(np.load(f))
            print(f"  seed {seed}: cached", flush=True)
            continue
        t0 = time.time()
        if model == "2s":
            pos = lytr > 0
            cp = {k: v for k, v in LGB.items() if k != "objective"}
            clf = lgb.LGBMClassifier(**cp, random_state=seed)
            clf.fit(Xtr, pos.astype(np.int8))
            pb = clf.predict_proba(Xev)[:, 1]
            reg = lgb.LGBMRegressor(**LGB, random_state=seed)
            reg.fit(Xtr[pos], lytr[pos])
            lp = pb * np.clip(reg.predict(Xev), 0, None)
        else:
            m = lgb.LGBMRegressor(**LGB, random_state=seed)
            m.fit(Xtr, lytr)
            lp = m.predict(Xev)
        np.save(f, lp.astype(np.float32))
        preds.append(lp)
        pv = lp[mask] if mask is not None else lp
        print(f"  seed {seed}: {rmsle(yev, np.expm1(np.clip(pv, 0, None))):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    n = len(preds)

    def score(vec):
        pv = vec[mask] if mask is not None else vec
        return rmsle(yev, np.expm1(np.clip(pv, 0, None)))

    print(f"\nsubset-averaged ensemble curve ({model}, {proto}):")
    for k in [1, 2, 3, 5, 7, 9]:
        combos = list(combinations(range(n), k))
        if len(combos) > 40:
            rng = np.random.default_rng(0)
            combos = [combos[i] for i in rng.choice(len(combos), 40, replace=False)]
        sc = np.array([score(np.mean([preds[i] for i in c], 0)) for c in combos])
        print(f"  k={k}: mean {sc.mean():.5f}  min {sc.min():.5f}  max {sc.max():.5f}")
    c9 = np.corrcoef(np.stack(preds))
    iu = np.triu_indices(n, 1)
    print(f"pairwise seed corr: mean {c9[iu].mean():.5f}  min {c9[iu].min():.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "l2",
         sys.argv[2] if len(sys.argv) > 2 else "SELECT")
