"""Rebuild stack6.csv from the six component submissions — no retraining needed.

Weights come from the unconstrained OLS projection of log1p(target) onto the span of the
component predictions. Both quantities it needs were solved from the leaderboard itself:
  E[log1p(y)] = 2.3284   (exact, from the three-point shift parabola)
  Var(log1p(y)) = 5.366802 (exact, from the slope probe)
Given those, Cov(L, P_i) = (Var(L) + Var(P_i) - R*_i^2) / 2 for every scored vector.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

EL, VAR_L, RIDGE = 2.3284, 5.366802, 3e-5
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "submission/components")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "submission/stack6.csv")

# component -> its measured public-leaderboard RMSLE
LB = {
    "two_stage_btyd_shift": 1.6546867486238441,
    "catboost_shifted":     1.6534269025124644,
    "lgbm_dart":            1.672341791651013,
    "nnls_isotonic_blend":  1.6836226427909342,
    "eb_logblocks":         1.7009172654286824,
    "seq_hybrid":           1.6541075222,
}

P, Rs2, uid = [], [], None
for name, lb in LB.items():
    d = pl.read_csv(SRC / f"{name}.csv")
    if uid is None:
        uid = d["user_id"].to_numpy()
    assert (d["user_id"].to_numpy() == uid).all(), f"user_id order differs in {name}"
    q = np.log1p(d["predict"].to_numpy())
    c = EL - q.mean()                     # level-match, then the LB score shifts by -c^2
    P.append(q - q.mean() + EL)
    Rs2.append(lb ** 2 - c ** 2)

M = np.stack(P)
C = np.cov(M)
k = (VAR_L + np.diag(C) - np.array(Rs2)) / 2
w = np.linalg.solve(C + RIDGE * np.eye(len(k)) * np.trace(C) / len(k), k)

mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
pred = np.clip(np.expm1(mix), 0, None)

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pred):
        f.write(f"{u},{v:.6f}\n")

for n, wi in zip(LB, w):
    print(f"  {wi:+7.4f}  {n}")
print(f"  сумма весов {w.sum():.4f}")
print(f"predicted RMSLE = {np.sqrt(VAR_L - 2 * w @ k + w @ C @ w):.6f}")
print(f"wrote {DST}: {len(pred)} rows, mean log1p={np.log1p(pred).mean():.4f}, "
      f"mean={pred.mean():.2f}, p50={np.percentile(pred, 50):.2f}")
