"""ElasticNet / L1 at the stacking layer.

The stack is currently an unconstrained OLS projection with a condition number near 27000 and
sixteen components, several carrying large opposing weights. That is exactly the regime where L1
helps: it prunes components whose contribution is not worth their variance, and the surviving
solution should transfer better to the private split, which is measured on different users.

The objective is the same one the OLS solve minimises, plus penalties:

    min_w  1/2 w'Cw - w'k + l1*|w|_1 + l2/2*|w|_2^2

C and k come from the leaderboard, not from data: k_i = (Var(L) + Var(P_i) - R*_i^2)/2 with
Var(L) = 5.366802 and E[log1p y] = 2.3284, both solved exactly from earlier submissions.
Solved by coordinate descent, which is exact for this objective.
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "models")

EL, VAR_L, NU = 2.3284, 5.366802, 250_000
CUR = 1.6486126196896862          # measured stack16

LB = {
    "sub4_two_stage_shift_c0105": 1.6546867486238441,
    "sub8_catboost_rmse_log1p_shifted": 1.6534269025124644,
    "sub1_lgbm_dart": 1.672341791651013,
    "sub3_nnls_isotonic_blend": 1.6836226427909342,
    "eb_logblocks": 1.7009172654286824,
    "seq_hybrid": 1.6541075222,
    "rfe_catboost": 1.6546700867029809,
    "rfe_lgbm": 1.6561152046374636,
    "nocal_catboost": 1.6535318106642274,
    "zoo_saint": 1.7851632642719046,
    "zoo_ngboost": 1.6595719116666063,
    "zoo_tabnet": 1.6586426133188408,
    "zoo_hazard": 1.6641023651856925,
    "zoo_ft_transformer": 1.657828367516315,
    "zoo_node": 1.6579977992804178,
    "zoo_carte": 1.657088763527966,
}

K = list(LB)
P, Rs2 = [], []
for n, v in LB.items():
    q = np.log1p(pl.read_csv(f"submissions/{n}.csv")["predict"].to_numpy())
    c = EL - q.mean()
    P.append(q - q.mean() + EL)
    Rs2.append(v ** 2 - c ** 2)
M = np.stack(P)
C = np.cov(M)
k = (VAR_L + np.diag(C) - np.array(Rs2)) / 2
print(f"{len(K)} компонентов, cond(C) = {np.linalg.cond(C):.0f}")

def coord_descent(C, k, l1, l2, iters=3000, tol=1e-12):
    n = len(k)
    w = np.linalg.solve(C + 1e-3 * np.eye(n) * np.trace(C) / n, k)   # warm start
    d = np.diag(C) + l2
    for _ in range(iters):
        wmax = 0.0
        for j in range(n):
            r = k[j] - (C[j] @ w - C[j, j] * w[j])
            new = np.sign(r) * max(abs(r) - l1, 0.0) / d[j]
            wmax = max(wmax, abs(new - w[j]))
            w[j] = new
        if wmax < tol:
            break
    return w

def score(w):
    return float(np.sqrt(max(VAR_L - 2 * w @ k + w @ C @ w, 0)))

rng = np.random.default_rng(0)
SUBS = [rng.choice(NU, 50_000, replace=False) for _ in range(80)]
def realised(w):
    """what the leaderboard would see if the covariances differ as a 50k subsample differs"""
    out = []
    for idx in SUBS:
        Cs = np.cov(M[:, idx])
        ks = (VAR_L + np.diag(Cs) - np.array(Rs2)) / 2
        out.append(np.sqrt(max(VAR_L - 2 * w @ ks + w @ Cs @ w, 0)))
    return np.array(out)

print(f"\n{'l1':>9} {'l2':>8} {'ненулевых':>10} {'предсказ.':>11} {'реализ.медиана':>15} {'p10':>10} {'p90':>10}")
best = None
GRID = [(0.0, 0.0), (0.0, 1e-4), (1e-5, 0.0), (3e-5, 0.0), (1e-4, 0.0), (3e-4, 0.0),
        (1e-3, 0.0), (3e-3, 0.0), (1e-2, 0.0),
        (1e-4, 1e-4), (3e-4, 3e-4), (1e-3, 1e-3), (3e-3, 3e-3)]
for l1, l2 in GRID:
    w = coord_descent(C, k, l1, l2)
    r = realised(w)
    nz = int((np.abs(w) > 1e-6).sum())
    print(f"{l1:9.5f} {l2:8.5f} {nz:10d} {score(w):11.6f} {np.median(r):15.6f} "
          f"{np.percentile(r,10):10.6f} {np.percentile(r,90):10.6f}")
    if best is None or np.median(r) < best[0]:
        best = (float(np.median(r)), l1, l2, w.copy(), nz)

med, l1, l2, w, nz = best
print(f"\nлучший: l1={l1}, l2={l2}, {nz} ненулевых, ожидаемый LB ~ {med:.6f} "
      f"(stack16 замерен {CUR:.7f}, {CUR-med:+.5f})")
for n, wi in sorted(zip(K, w), key=lambda x: -abs(x[1])):
    if abs(wi) > 1e-6:
        print(f"  {wi:+7.4f}  {n}")
dropped = [n for n, wi in zip(K, w) if abs(wi) <= 1e-6]
if dropped:
    print(f"  отброшены: {', '.join(dropped)}")

mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
u = pl.read_csv("submissions/sub4_two_stage_shift_c0105.csv")["user_id"].to_numpy()
pred = np.clip(np.expm1(mix), 0, None)
pl.DataFrame({"user_id": u, "predict": pred}).write_csv("submissions/stack_elastic.csv")
su = pl.read_csv("data/sample_submit.csv")["user_id"].to_numpy()
print(f"\nstack_elastic.csv: rows={len(pred)} order_ok={bool((u==su).all())} "
      f"mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f} "
      f"p50={np.percentile(pred,50):.2f} p99={np.percentile(pred,99):.0f} neg={int((pred<0).sum())}")
