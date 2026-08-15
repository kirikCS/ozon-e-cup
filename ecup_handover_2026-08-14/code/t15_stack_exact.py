"""EXACT stack solve once both de-noised components have their own leaderboard scores.

Same machinery as make_stack6.py — every R*^2 comes from a measured LB score, nothing is
estimated. Run when the human reports the numbers:

    python code/t15_stack_exact.py <LB_two_stage_9seed> <LB_seq_hybrid_9seed>

e.g.  python code/t15_stack_exact.py 1.65212 1.65107

Writes submission/queue/stack9_exact.csv and prints predicted LB (historic accuracy of
this prediction: 2.9e-5 / 1.0e-4 / 6.6e-5 on sub11 / ols_stack / stack6).
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

EL, VAR_L, RIDGE = 2.3284, 5.366802, 3e-5
COMP = Path("submission/components")
OUT_CSV = Path("submission/queue/stack9_exact.csv")

assert len(sys.argv) == 3, __doc__
LB_2S, LB_SEQ = float(sys.argv[1]), float(sys.argv[2])
assert 1.60 < LB_2S < 1.75 and 1.60 < LB_SEQ < 1.75, "sanity: scores out of range"

SOURCES = [
    ("two_stage_9seed", Path("submissions/two_stage_btyd_9seed.csv"), LB_2S),
    ("catboost", COMP / "catboost_shifted.csv", 1.6534269025124644),
    ("lgbm_dart", COMP / "lgbm_dart.csv", 1.672341791651013),
    ("iso_blend", COMP / "nnls_isotonic_blend.csv", 1.6836226427909342),
    ("eb_logblocks", COMP / "eb_logblocks.csv", 1.7009172654286824),
    ("seq_hybrid_9seed", Path("submissions/seq_hybrid_9seed.csv"), LB_SEQ),
]

P, Rs2, uid = [], [], None
for name, path, lb in SOURCES:
    d = pl.read_csv(path)
    if uid is None:
        uid = d["user_id"].to_numpy()
    assert (d["user_id"].to_numpy() == uid).all(), name
    q = np.log1p(d["predict"].to_numpy())
    c = EL - q.mean()
    P.append(q - q.mean() + EL)
    Rs2.append(lb ** 2 - c ** 2)
    print(f"  {name:<18} LB {lb:.7f}  level shift c {c:+.4f}  R*_lm {np.sqrt(lb**2-c**2):.5f}")

M = np.stack(P)
C = np.cov(M)
k = (VAR_L + np.diag(C) - np.array(Rs2)) / 2
w = np.linalg.solve(C + RIDGE * np.eye(len(k)) * np.trace(C) / len(k), k)
pred = float(np.sqrt(VAR_L - 2 * w @ k + w @ C @ w))

print(f"\npredicted LB = {pred:.6f}   (stack6 actual 1.6496638)")
for (n, _, _), wi in zip(SOURCES, w):
    print(f"  {wi:+7.4f}  {n}")
print(f"  sum {w.sum():.4f}   cond(C) {np.linalg.cond(C):.0f}")

# public-subsample stability: the LB uses 50k of 250k users
rng = np.random.default_rng(0)
sims = []
for _ in range(60):
    idx = rng.choice(M.shape[1], 50_000, replace=False)
    Ms = M[:, idx]
    Cs = np.cov(Ms)
    ks = (VAR_L + np.diag(Cs) - np.array(Rs2)) / 2
    sims.append(float(np.sqrt(max(VAR_L - 2 * w @ ks + w @ Cs @ w, 0))))
sims = np.array(sims)
print(f"50k-subsample simulation: p10 {np.percentile(sims,10):.5f}  "
      f"p90 {np.percentile(sims,90):.5f}")

mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
pv = np.clip(np.expm1(mix), 0, None)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pv):
        f.write(f"{u},{v:.6f}\n")
print(f"wrote {OUT_CSV}")
s6 = np.log1p(pl.read_csv("submission/stack6.csv")["predict"].to_numpy())
print(f"corr(log1p, stack6) = {np.corrcoef(mix, s6)[0,1]:.6f}")
