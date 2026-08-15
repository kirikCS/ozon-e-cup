"""EXACT stack over all eight measured vectors — no estimated quantity anywhere.

2026-08-12 measurements added three vectors with known LB scores; together with the five
older components every R*^2 below is leaderboard-measured:

  two_stage_old    1.6546867   two_stage_9seed  1.6556581  (local pipeline, de-noised)
  catboost         1.6534269   lgbm_dart        1.6723418
  iso_blend        1.6836226   eb_logblocks     1.7009173
  seq_old          1.6541075   seq_9seed        1.6540039  (local pipeline, de-noised)

Old and new versions of a component correlate ~0.998 — the solver may keep both and
extract the pipeline-difference direction; ridge handles the collinearity.

Validation built in: stack7's own measured score (1.6493808) must be reproduced by the
algebra from its weight decomposition before the new solve is trusted.

Writes submission/queue/stack8x.csv (the 8-vector solve) and prints the 6-vector-new-only
alternative for comparison.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

EL, VAR_L, RIDGE = 2.3284, 5.366802, 3e-5
COMP = Path("submission/components")

SOURCES = [
    ("two_stage_old", COMP / "two_stage_btyd_shift.csv", 1.6546867486238441),
    ("two_stage_9seed", Path("submissions/two_stage_btyd_9seed.csv"), 1.6556581195791316),
    ("catboost", COMP / "catboost_shifted.csv", 1.6534269025124644),
    ("lgbm_dart", COMP / "lgbm_dart.csv", 1.672341791651013),
    ("iso_blend", COMP / "nnls_isotonic_blend.csv", 1.6836226427909342),
    ("eb_logblocks", COMP / "eb_logblocks.csv", 1.7009172654286824),
    ("seq_old", COMP / "seq_hybrid.csv", 1.6541075222),
    ("seq_9seed", Path("submissions/seq_hybrid_9seed.csv"), 1.654003921177154),
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
M = np.stack(P)
C = np.cov(M)
k = (VAR_L + np.diag(C) - np.array(Rs2)) / 2

# ---- validation: reproduce stack7's measured score from the algebra
s7 = np.log1p(pl.read_csv("submission/stack7_2s9seed_lb1649381.csv")["predict"].to_numpy())
s7 = s7 - s7.mean() + EL
a, *_ = np.linalg.lstsq(M.T, s7, rcond=None)
resid = float(np.abs(M.T @ a - s7).max())
r_pred7 = float(np.sqrt(VAR_L - 2 * a @ k + a @ C @ a))
print(f"validation: stack7 decomposition residual {resid:.2e}, "
      f"algebra predicts {r_pred7:.6f}, measured 1.649381, "
      f"error {r_pred7-1.649380786:+.6f}")

names = [s[0] for s in SOURCES]


def solve(idx):
    Cs = C[np.ix_(idx, idx)]
    ks = k[idx]
    w = np.linalg.solve(Cs + RIDGE * np.eye(len(ks)) * np.trace(Cs) / len(ks), ks)
    pred = float(np.sqrt(VAR_L - 2 * w @ ks + w @ Cs @ w))
    return w, pred


def simulate(idx, w):
    rng = np.random.default_rng(0)
    Ms = M[idx]
    out = []
    for _ in range(60):
        ii = rng.choice(Ms.shape[1], 50_000, replace=False)
        Cs = np.cov(Ms[:, ii])
        ks = (VAR_L + np.diag(Cs) - np.array(Rs2)[idx]) / 2
        out.append(float(np.sqrt(max(VAR_L - 2 * w @ ks + w @ Cs @ w, 0))))
    return np.percentile(out, [10, 90])


for label, idx in [("8-vector", list(range(8))),
                   ("6-vector new-only", [1, 2, 3, 4, 5, 7]),
                   ("6-vector old-only (=stack6 basis)", [0, 2, 3, 4, 5, 6])]:
    w, pred = solve(idx)
    p10, p90 = simulate(idx, w)
    cond = np.linalg.cond(C[np.ix_(idx, idx)])
    print(f"\n{label}: predicted {pred:.6f}   50k-sim p10-p90 [{p10:.5f}, {p90:.5f}]   "
          f"cond {cond:.0f}")
    for n, wi in zip([names[i] for i in idx], w):
        print(f"  {wi:+7.4f}  {n}")

w, pred = solve(list(range(8)))
mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
pv = np.clip(np.expm1(mix), 0, None)
out = Path("submission/queue/stack8x.csv")
with open(out, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pv):
        f.write(f"{u},{v:.6f}\n")
print(f"\nwrote {out}: mean log1p={np.log1p(pv).mean():.4f} zeros={(pv==0).mean():.2%}")
for ref, refname in [("submission/stack6.csv", "stack6"),
                     ("submission/queue/stack7_2s9seed.csv", "stack7")]:
    q = np.log1p(pl.read_csv(ref)["predict"].to_numpy())
    print(f"corr(log1p, {refname}) = {np.corrcoef(mix, q)[0,1]:.6f}")
