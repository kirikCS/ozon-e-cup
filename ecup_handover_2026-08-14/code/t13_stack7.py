"""Candidate stack: stack6 with the two-stage component replaced by its 9-seed average.

Same solve as make_stack6.py (OLS via LB-derived covariances), two changes:
  * the component vector two_stage_btyd_shift.csv -> submissions/two_stage_btyd_9seed.csv
  * its R*^2 is not LB-measured (the averaged vector was never submitted) but predicted:
      R*_new^2 = R*_old_lm^2 - (probe_seed42^2 - probe_ens9^2)
    where the probe terms come from t12 on the shipped PROBE geometry. Seed noise is
    orthogonal to the target, so averaging removes variance without touching Cov(L, P) —
    exactly the change the parabola algebra can price.

Prints the predicted LB, the weight table, and the sensitivity of the prediction to the
R*_new estimate (+-0.001 sweep). Writes submission/queue/stack7_2s9seed.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

EL, VAR_L, RIDGE = 2.3284, 5.366802, 3e-5
COMP = Path("submission/components")
OUT_CSV = Path("submission/queue/stack7_2s9seed.csv")

probe = json.load(open("model_out/t12_probe.json"))
R_OLD_LM = 1.6546852                      # sub4 level-matched (text.md §7.1)
d2 = probe["probe_seed42"] ** 2 - probe["probe_ens9"] ** 2
R_NEW = float(np.sqrt(R_OLD_LM ** 2 - d2))
print(f"probe: seed42 {probe['probe_seed42']:.5f} ens9 {probe['probe_ens9']:.5f}"
      f"  -> R*_new {R_NEW:.5f} (old {R_OLD_LM})")

SOURCES = [
    ("two_stage_9seed", Path("submissions/two_stage_btyd_9seed.csv"), None),
    ("catboost_shifted", COMP / "catboost_shifted.csv", 1.6534269025124644),
    ("lgbm_dart", COMP / "lgbm_dart.csv", 1.672341791651013),
    ("nnls_isotonic_blend", COMP / "nnls_isotonic_blend.csv", 1.6836226427909342),
    ("eb_logblocks", COMP / "eb_logblocks.csv", 1.7009172654286824),
    ("seq_hybrid", COMP / "seq_hybrid.csv", 1.6541075222),
]


def solve(r_new):
    P, Rs2, uid = [], [], None
    for name, path, lb in SOURCES:
        d = pl.read_csv(path)
        if uid is None:
            uid = d["user_id"].to_numpy()
        assert (d["user_id"].to_numpy() == uid).all(), name
        q = np.log1p(d["predict"].to_numpy())
        c = EL - q.mean()
        P.append(q - q.mean() + EL)
        Rs2.append(r_new ** 2 if lb is None else lb ** 2 - c ** 2)
    M = np.stack(P)
    C = np.cov(M)
    k = (VAR_L + np.diag(C) - np.array(Rs2)) / 2
    w = np.linalg.solve(C + RIDGE * np.eye(len(k)) * np.trace(C) / len(k), k)
    pred_R = float(np.sqrt(VAR_L - 2 * w @ k + w @ C @ w))
    return w, pred_R, M, uid


w, pred_R, M, uid = solve(R_NEW)
print(f"\npredicted LB = {pred_R:.6f}   (stack6 predicted 1.649598, actual 1.6496638)")
for (n, _, _), wi in zip(SOURCES, w):
    print(f"  {wi:+7.4f}  {n}")
print(f"  sum {w.sum():.4f}")

print("\nsensitivity to R*_new:")
for dr in [-0.001, -0.0005, 0.0, 0.0005, 0.001]:
    _, r, _, _ = solve(R_NEW + dr)
    print(f"  R*_new {R_NEW+dr:.5f} -> predicted {r:.6f}")

mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
pred = np.clip(np.expm1(mix), 0, None)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"\nwrote {OUT_CSV}: mean log1p={np.log1p(pred).mean():.4f} mean={pred.mean():.2f}")

# corr with stack6 for the record
s6 = pl.read_csv("submission/stack6.csv")
q6 = np.log1p(s6["predict"].to_numpy())
print(f"corr(log1p stack7, log1p stack6) = {np.corrcoef(mix, q6)[0,1]:.6f}")
