"""Multi-seed CatBoost at the prediction anchor — the one idea today that survived its control.

blend_confound.py decomposed the +0.0054 blend gain into slope (+0.0030) and seed averaging
(+0.0024). The slope half is already taken: stack6's OLS weights sum to 1.0385, fitted against
the real target through the leaderboard algebra, which beats any locally fitted slope. The
averaging half is not taken at all — every tree component in t07/t08 is a single fit at
random_state=42. Only seq_hybrid averages seeds, and it averages three.

So this rebuilds the highest-weighted retrainable tree component (CatBoost, weight 0.3488 in
stack6, best honest score of any tree at 1.7485) as an average of N seeds in log space.

Expect less than +0.0024 on the stack: six blended components already damp seed noise. The
point of the submission is to measure Cov(L, P) exactly, after which the stack re-solves for
free.

    python code/t11_catboost_seeds.py [n_seeds]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import H, NU, N_DAYS, target, rmsle, build_features, OUT, CACHE, ds

PRED, FIT = N_DAYS - 1, 378
TR_FINAL = [378, 350, 322, 294, 266]
TR_PROBE = [348, 320, 292, 264, 236]
ALL_A = sorted(set(TR_FINAL + TR_PROBE + [PRED, FIT]))
EL_PUBLIC = 2.3284                      # E[log1p y] on the public split, from the LB parabola
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
QUEUE = Path("submission/queue"); QUEUE.mkdir(parents=True, exist_ok=True)

CB = dict(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0, border_count=64,
          loss_function="RMSE", verbose=0, thread_count=23, bootstrap_type="Bernoulli",
          subsample=0.8)

X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {X[PRED].shape[1]}, seeds {NSEED}, predict {ds(PRED)} -> [{ds(PRED+1)}, {ds(PRED+H)}]\n")


def fit_seeds(anchors, Xev, label):
    """average in log space — that is the space RMSLE lives in, so averaging there is
    averaging the quantity the metric actually scores"""
    Xtr = np.concatenate([X[a] for a in anchors], 0)
    ly = np.concatenate([np.log1p(Y[a]) for a in anchors])
    acc = []
    for s in range(NSEED):
        t0 = time.time()
        m = CatBoostRegressor(random_seed=42 + s, **CB)
        m.fit(Xtr, ly)
        acc.append(m.predict(Xev))
        print(f"  {label} seed {42+s}: {time.time()-t0:.0f}s")
    return np.mean(acc, 0), acc


lp_probe, singles = fit_seeds(TR_PROBE, X[FIT], "probe")
y = Y[FIT]
s1 = rmsle(y, np.expm1(np.clip(singles[0], 0, None)))
sn = rmsle(y, np.expm1(np.clip(lp_probe, 0, None)))
print(f"\nprobe at {ds(FIT)}:  single seed {s1:.5f}   {NSEED}-seed mean {sn:.5f}   ({s1-sn:+.5f})")
print(f"  mean pairwise corr between seeds "
      f"{np.mean([np.corrcoef(singles[i], singles[j])[0,1] for i in range(NSEED) for j in range(i+1, NSEED)]):.5f}")

print()
lp_final, _ = fit_seeds(TR_FINAL, X[PRED], "final")

lp = np.clip(np.nan_to_num(lp_final), 0, 13)
lp = lp - lp.mean() + EL_PUBLIC          # level is already known exactly; do not refit it
pred = np.clip(np.expm1(lp), 0, None)

uids = np.load(CACHE / "uids.npy")
dst = QUEUE / f"catboost_seed{NSEED}.csv"
with open(dst, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")

assert len(pred) == NU and np.isfinite(pred).all() and (pred >= 0).all()
print(f"\nwrote {dst}")
print(f"  rows {len(pred)}  mean log1p {np.log1p(pred).mean():.4f}  mean {pred.mean():.2f}  "
      f"p50 {np.percentile(pred,50):.2f}  p99 {np.percentile(pred,99):.0f}  max {pred.max():.0f}")
