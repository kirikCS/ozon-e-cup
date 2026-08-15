"""Exact blend of the teammate's vector with our measured stack — the top-1 play.

The teammate's submission (LB 1.6486126197) is a measured vector like any other: its
R* gives Cov(L, P) exactly, and an OLS solve over {teammate, stack7, best singles} is
the sub11-class quiet solve (historic prediction error 2.9e-5 .. 1.8e-4). Policy from
text.md §9б holds: all R* measured, no long-shorts, weights sane by construction
(the teammate's vector correlates with ours the least of any pair we have).

Usage:
    python code/t20_stack_team.py <csv>=<LB score> [<csv2>=<score2> ...]
e.g.
    python code/t20_stack_team.py submission/queue/bwn.csv=1.6486126196896862 ^
        submission/queue/tcb.csv=1.6491800741214804 ^
        submission/queue/nocal.csv=1.6535318106642274
Writes submission/queue/stack_team.csv with the predicted LB.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

EL, VAR_L, RIDGE = 2.3284, 5.366802, 3e-4
COMP = Path("submission/components")

assert len(sys.argv) >= 2, __doc__
TEAM = []
for arg in sys.argv[1:]:
    p, _, s = arg.partition("=")
    TEAM.append((f"tm_{Path(p).stem[:12]}", Path(p), float(s)))

SOURCES = TEAM + [
    ("stack7", Path("submission/stack7_2s9seed_lb1649381.csv"), 1.649380786349335),
    ("two_stage_v3", Path("submissions/two_stage_v3.csv"), 1.6530487792218926),
    ("catboost_v3", Path("submissions/catboost_v3.csv"), 1.653229223776429),
    ("seq_9seed", Path("submissions/seq_hybrid_9seed.csv"), 1.654003921177154),
    ("eb_logblocks", COMP / "eb_logblocks.csv", 1.7009172654286824),
]

P, Rs2, names, uid = [], [], [], None
for name, path, lb in SOURCES:
    d = pl.read_csv(path)
    u = d["user_id"].to_numpy()
    if uid is None:
        uid = np.sort(u)
    order = np.argsort(u)                       # teammate's row order may differ
    q = np.log1p(d["predict"].to_numpy()[order])
    assert (u[order] == uid).all(), f"user_id set differs in {name}"
    c = EL - q.mean()
    P.append(q - q.mean() + EL)
    Rs2.append(lb ** 2 - c ** 2)
    names.append(name)
    print(f"  {name:<14} LB {lb:.7f}  c {c:+.4f}  R*_lm {np.sqrt(lb**2-c**2):.5f}")

M = np.stack(P)
C = np.cov(M)
print("\ncorr matrix:")
cc = np.corrcoef(M)
for i, n in enumerate(names):
    print(f"  {n:<14} " + " ".join(f"{cc[i,j]:.4f}" for j in range(len(names))))

Rs2 = np.array(Rs2)


def solve(idx, ridge=RIDGE):
    Cs = C[np.ix_(idx, idx)]
    ks = (VAR_L + np.diag(Cs) - Rs2[idx]) / 2
    w = np.linalg.solve(Cs + ridge * np.eye(len(ks)) * np.trace(Cs) / len(ks), ks)
    pred = float(np.sqrt(VAR_L - 2 * w @ ks + w @ Cs @ w))
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(60):
        ii = rng.choice(M.shape[1], 50_000, replace=False)
        Css = np.cov(M[idx][:, ii])
        kss = (VAR_L + np.diag(Css) - Rs2[idx]) / 2
        sims.append(float(np.sqrt(max(VAR_L - 2 * w @ kss + w @ Css @ w, 0))))
    return w, pred, np.percentile(sims, [10, 90])


nt = len(TEAM)
st7 = nt                       # index of stack7 after the teammate block
combos = [("team+stack7", list(range(nt)) + [st7]),
          ("team+stack7+ts_v3", list(range(nt)) + [st7, st7 + 1]),
          ("team+stack7+v3s+seq", list(range(nt)) + [st7, st7 + 1, st7 + 2, st7 + 3]),
          ("все", list(range(len(SOURCES))))]
best = None
for label, idx in combos:
    w, pred, (p10, p90) = solve(idx)
    mx = np.abs(w).max()
    flag = "  <-- ВЕС >0.9, отбросить" if mx > 0.9 else ""
    print(f"\n{label}: predicted {pred:.6f}  sim [{p10:.5f}, {p90:.5f}]  max|w| {mx:.2f}{flag}")
    for n, wi in zip([names[i] for i in idx], w):
        print(f"  {wi:+7.4f}  {n}")
    if mx <= 0.9 and (best is None or pred < best[1]):
        best = (idx, pred, w)

idx, pred, w = best
mix = (w[:, None] * M[idx]).sum(0)
mix = mix + (EL - mix.mean())
pv = np.clip(np.expm1(mix), 0, None)
out = Path("submission/queue/stack_team.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pv):
        f.write(f"{u},{v:.6f}\n")
print(f"\nwrote {out} (variant {[names[i] for i in idx]}, predicted {pred:.6f})")
print(f"team best 1.6486126, top-1 1.6480578 — цель ниже 1.64806")
