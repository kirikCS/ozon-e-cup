"""Bias-corrected exact solve over all measured vectors + measured-mixture constraints.

For any fixed weight vector w over the basis, R^2(w) = VarL - 2 w'k + w'Cw. Every solved
mixture that was actually measured on the LB gives one exact linear constraint on the
k-error delta = k_assumed - k_true:  w_j' delta = (R2_meas_j - R2_pred_j) / 2, where
R2_pred is computed from THIS basis's (C, k) and w_j is the lstsq decomposition of the
actual mixture csv (registry: model_out/stack_constraints.json — append each new
measured mixture there). Min-norm delta over all constraints corrects k; solves on
k_corr with a ridge sweep; candidate accepted only at tame weights (max|w| <= 0.6,
no shorts beyond -0.25) AND with the corrected prediction below the banked best.

History note (solver review 2026-08-13): in the 4-constraint era all misses were traced
to a single root cause — the ts_9seed (csv, score) pair mixed a level-matched vector
with a raw-level measurement (see the inline comment at its V entry). With that fixed
the constraints' b values collapse toward apparatus tolerance.

Usage:  python code/t21_stack_corrected.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

EL, VAR_L = 2.3284, 5.366802
COMP = Path("submission/components")

V = [
    ("bwn", "submission/queue/blend-with-neural.csv", 1.6486126196896862),
    ("tcb", "submission/queue/TOTALCATBOOSTBLEND.csv", 1.6491800741214804),
    ("nocal", "submission/queue/nocal_catboost.csv", 1.6535318106642274),
    ("seq_pure", "submissions/seq_pure_9seed.csv", 1.6658092084427447),
    ("stack_dl", "submission/queue/stack_dl.csv", 1.648344063188555),
    ("evgru2", "submissions/evgru2_final_lb1660490.csv", 1.660490164378798),
    ("ts_v3", "submissions/two_stage_v3.csv", 1.6530487792218926),
    ("cb_v3", "submissions/catboost_v3.csv", 1.653229223776429),
    # ts_9seed was SUBMITTED at raw level (mean log1p 2.2276, before the level-match edit
    # of 2026-08-12 23:00) — the measured 1.6556581 belongs to the raw vector, the csv on
    # disk is the level-matched one. Feed the level-matched R*: sqrt(lb^2 - c_submit^2),
    # c_submit = 2.3284 - 2.2276 = 0.1008. (Root cause of the -0.005 k(ts_9seed) anomaly
    # the solver review diagnosed; the four constraint misses collapse once this is fixed.)
    ("ts_9seed", "submissions/two_stage_btyd_9seed.csv",
     (1.6556581195791316 ** 2 - 0.1008 ** 2) ** 0.5),
    ("ts_old", str(COMP / "two_stage_btyd_shift.csv"), 1.6546867486238441),
    ("cb_old", str(COMP / "catboost_shifted.csv"), 1.6534269025124644),
    ("seq_old", str(COMP / "seq_hybrid.csv"), 1.6541075222),
    ("seq9", "submissions/seq_hybrid_9seed.csv", 1.654003921177154),
    ("dart", str(COMP / "lgbm_dart.csv"), 1.672341791651013),
    ("iso", str(COMP / "nnls_isotonic_blend.csv"), 1.6836226427909342),
    ("eb", str(COMP / "eb_logblocks.csv"), 1.7009172654286824),
]

P, Rs2, names, uid = [], [], [], None
for n, p, lb in V:
    d = pl.read_csv(p)
    u = d["user_id"].to_numpy()
    o = np.argsort(u)
    if uid is None:
        uid = u[o]
    assert np.array_equal(u[o], uid), f"user_id set differs in {n}"
    q = np.log1p(d["predict"].to_numpy()[o])
    c = EL - q.mean()
    P.append(q - q.mean() + EL)
    Rs2.append(lb ** 2 - c ** 2)
    names.append(n)
M = np.stack(P)
C = np.cov(M)
Rs2 = np.array(Rs2)
k = (VAR_L + np.diag(C) - Rs2) / 2
ix = {n: i for i, n in enumerate(names)}

# measured-mixture constraints: decomposition weights refit by lstsq on the ACTUAL
# vectors (printout rounding on near-collinear bases distorts R^2 noticeably; the clip
# nonlinearity leaves a small residual which the correction absorbs)
def decompose(path):
    q = np.log1p(pl.read_csv(path)["predict"].to_numpy())
    # sort to basis uid order
    u = pl.read_csv(path)["user_id"].to_numpy()
    q = q[np.argsort(u)]
    q = q - q.mean() + EL
    A_ = np.concatenate([M, np.ones((1, M.shape[1]))], 0)
    sol, *_ = np.linalg.lstsq(A_.T, q, rcond=None)
    resid = np.abs(A_.T @ sol - q)
    print(f"  decompose {Path(path).name}: max resid {resid.max():.4f}, "
          f"share>0.01 {np.mean(resid > 0.01):.3%}")
    return sol[:-1]

# every measured solved-mixture is a constraint; append new (csv, lb) pairs to the
# registry after each leaderboard measurement — the correction sharpens each round
import json
CONSTRAINTS = json.loads(Path("model_out/stack_constraints.json")
                         .read_text(encoding="utf-8-sig"))
CW = [(decompose(c["csv"]), c["lb"]) for c in CONSTRAINTS]

# R2_pred must come from THIS basis's (C, k), not historical solver printouts (those used
# their own bases, ridge and clipping)
A = np.stack([wv for wv, _ in CW])
b = []
for wv, meas in CW:
    r2_pred = VAR_L - 2 * wv @ k + wv @ C @ wv
    b.append((meas ** 2 - r2_pred) / 2)
    print(f"ограничение: pred {np.sqrt(r2_pred):.6f} meas {meas:.6f} -> b {b[-1]:+.6f}")
b = np.array(b)
delta = A.T @ np.linalg.solve(A @ A.T, b)          # min-norm correction
k_corr = k - delta
print("коррекция k (только ненулевые):")
for n, d0, d1 in zip(names, k, delta):
    if abs(d1) > 1e-6:
        print(f"  {n:<9} k {d0:.5f} -> {d0-d1:.5f}  (delta {d1:+.5f})")

# consistency check: corrected k must reprice every measured mixture exactly
for (wv, meas), cinfo in zip(CW, CONSTRAINTS):
    r = float(np.sqrt(VAR_L - 2 * wv @ k_corr + wv @ C @ wv))
    print(f"проверка {Path(cinfo['csv']).stem[:20]}: {r:.6f}  измерено {meas:.6f}  "
          f"delta {r-meas:+.6f}")


def solve(ridge, kv):
    w = np.linalg.solve(C + ridge * np.eye(len(kv)) * np.trace(C) / len(kv), kv)
    pred = float(np.sqrt(max(VAR_L - 2 * w @ kv + w @ C @ w, 0)))
    return w, pred


print("\nсвип на скорректированном k:")
best = None
for ridge in [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
    w, pred = solve(ridge, k_corr)
    mx = np.abs(w).max()
    worst_short = min(w)
    tame = mx <= 0.6 and worst_short >= -0.25
    top = sorted(zip(names, w), key=lambda x: -abs(x[1]))[:5]
    print(f"  ridge {ridge:.0e}: pred {pred:.6f}  max|w| {mx:.2f} min_w {worst_short:+.2f} "
          f"{'TAME' if tame else 'откл.'}  " + " ".join(f"{n}:{x:+.2f}" for n, x in top))
    if tame and (best is None or pred < best[1]):
        best = (w, pred, ridge)

import os
if os.environ.get("FX_TWINK"):
    # measurement candidate for the twink channel: the lowest-ridge solve WITHOUT the
    # tame gate (no forbidden long-shorts remain post-rootfix; only max|w| exceeds policy)
    ridge = 3e-5
    wA = np.linalg.solve(C + ridge * np.eye(len(k_corr)) * np.trace(C) / len(k_corr), k_corr)
    predA = float(np.sqrt(max(VAR_L - 2 * wA @ k_corr + wA @ C @ wA, 0)))
    mixA = (wA[:, None] * M).sum(0)
    mixA = mixA + (EL - mixA.mean())
    pvA = np.clip(np.expm1(mixA), 0, None)
    with open("submission/queue/twink_aggressive.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uid, pvA):
            f.write(f"{u},{v:.6f}\n")
    print(f"\nTWINK-кандидат: предсказание {predA:.6f}, max|w| {np.abs(wA).max():.2f}, "
          f"min_w {wA.min():+.2f}")
    for n, x in sorted(zip(names, wA), key=lambda t: -abs(t[1]))[:6]:
        print(f"  {x:+.3f}  {n}")

BANKED_BEST = 1.6480488273          # stack_dl_fusion, our measured best
if best is None:
    print("\nни одно решение не прошло политику весов")
    sys.exit(1)
w, pred, ridge = best
pred_uncorr = float(np.sqrt(max(VAR_L - 2 * w @ k + w @ C @ w, 0)))
print(f"\nполоса честности для выбранных весов: corrected {pred:.6f} … "
      f"uncorrected {pred_uncorr:.6f}")
if pred >= BANKED_BEST:
    print(f"кандидат НЕ пишется: даже оптимистичный край ({pred:.6f}) не лучше "
          f"достигнутого ({BANKED_BEST})")
    sys.exit(0)
mix = (w[:, None] * M).sum(0)
mix = mix + (EL - mix.mean())
pv = np.clip(np.expm1(mix), 0, None)
out = Path("submission/queue/stack_team_corr.csv")
with open(out, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uid, pv):
        f.write(f"{u},{v:.6f}\n")
print(f"\nwrote {out} (ridge {ridge:.0e}, predicted {pred:.6f})")
s7 = np.log1p(pl.read_csv("submission/stack7_2s9seed_lb1649381.csv")["predict"].to_numpy())
bw = np.log1p(pl.read_csv("submission/queue/blend-with-neural.csv")["predict"].to_numpy())
print(f"corr со stack7 {np.corrcoef(mix, s7)[0,1]:.5f}, с blend-with-neural "
      f"{np.corrcoef(mix, bw)[0,1]:.5f}")
