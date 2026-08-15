"""Does transductive pseudo-labelling stack with seed-ensembling?

fx_pseudo measured -0.001 for a single seed vs single seed. If part of that is implicit
variance reduction (pulling toward a stable prior), the gain on top of a 9-seed ensemble
is smaller. This measures ens9(pseudo-students) vs ens9(plain students) on all three
protocols, teacher = plain ens9 of the same protocol, w=0.5.

Usage:  python -u code/fx_pseudo_ens.py SELECT
        python -u code/fx_pseudo_ens.py VAL
        python -u code/fx_pseudo_ens.py HON
"""
import sys
import time
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
W = 0.5

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True, objective="regression")


def main(proto):
    TR, EV, use_mask = PROTO[proto]
    anchors = sorted(set(TR + [EV]))
    X, names = build_features(anchors)
    lytr = np.concatenate([np.log1p(target(a)) for a in TR])
    yev = target(EV)
    mask = cohort_rule(EV) if use_mask else None
    yv = yev[mask] if mask is not None else yev
    Xtr = np.concatenate([X[a] for a in TR], 0)
    Xev = X[EV]

    def score(vec):
        pv = vec[mask] if mask is not None else vec
        return rmsle(yv, np.expm1(np.clip(pv, 0, None)))

    # plain seed vectors (cache from fx_seed_ens; legacy SELECT names lack the proto tag)
    plain = []
    for s in SEEDS:
        for f in [OUT / f"fx_se_l2_{proto}_s{s}.npy"] + (
                [OUT / f"fx_se_l2_s{s}.npy"] if proto == "SELECT" else []):
            if f.exists():
                plain.append(np.load(f))
                break
        else:
            t0 = time.time()
            m = lgb.LGBMRegressor(**LGB, random_state=s)
            m.fit(Xtr, lytr)
            lp = m.predict(Xev)
            np.save(OUT / f"fx_se_l2_{proto}_s{s}.npy", lp.astype(np.float32))
            plain.append(lp)
            print(f"  plain seed {s}: {score(lp):.5f} ({time.time()-t0:.0f}s)", flush=True)
    teacher = np.mean(plain, 0)
    print(f"{proto}: plain ens9 {score(teacher):.5f}  mean single "
          f"{np.mean([score(p) for p in plain]):.5f}", flush=True)

    Xa = np.concatenate([Xtr, Xev], 0)
    ya = np.concatenate([lytr, teacher])
    sw = np.concatenate([np.ones(len(lytr)), np.full(len(teacher), W)])
    ps = []
    for s in SEEDS:
        t0 = time.time()
        m = lgb.LGBMRegressor(**LGB, random_state=s)
        m.fit(Xa, ya, sample_weight=sw)
        lp = m.predict(Xev)
        ps.append(lp)
        print(f"  pseudo seed {s}: {score(lp):.5f} ({time.time()-t0:.0f}s)", flush=True)
    ens_p = np.mean(ps, 0)
    sp, s0 = score(ens_p), score(teacher)
    print(f"{proto}: pseudo ens9 {sp:.5f}  vs plain ens9 {s0:.5f}  delta {sp-s0:+.5f}",
          flush=True)
    np.save(OUT / f"fx_pseudo_ens_{proto}.npy", ens_p.astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SELECT")
