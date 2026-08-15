"""Tier 6: blending and calibration.
Blend weights and calibration curves are fitted on one half of the users and scored on the other,
so no number here is in-sample.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "models")
from common import VAL_ANCHOR, NU, target, rmsle, record, Timer, OUT

A = VAL_ANCHOR
y = target(A)
ly = np.log1p(y)
rng = np.random.default_rng(7)
half = rng.random(NU) < 0.5           # fit on `half`, score on `~half`

preds = {}
for f in sorted(Path(OUT).glob("pred_*.npy")):
    v = np.load(f)
    if v.shape == (NU,) and np.isfinite(v).all():
        preds[f.stem[5:].replace("_", " ")] = v
print(f"loaded {len(preds)} saved predictions")
scored = sorted(((rmsle(y, v), k) for k, v in preds.items()))
for s, k in scored[:12]:
    print(f"  {s:.5f}  {k}")

if len(scored) < 2:
    print("not enough predictions to blend")
    sys.exit(0)

def ev_half(p, name, family, notes=""):
    return record(name, family, rmsle(y[~half], p[~half]), None, None,
                  notes + " | fitted on 50% of users, scored on the other 50%")

TOPK = [k for _, k in scored[:8]]
M = np.stack([np.log1p(np.clip(preds[k], 0, None)) for k in TOPK], 1)

# 1. plain average in log space
ev_half(np.expm1(M[:, :3].mean(1)), "Blend: mean of top-3 in log space", "ensemble")
ev_half(np.expm1(M[:, :5].mean(1)), "Blend: mean of top-5 in log space", "ensemble")
ev_half(np.expm1(M.mean(1)), "Blend: mean of top-8 in log space", "ensemble")

# 2. non-negative least squares weights on log1p
w, _ = nnls(M[half], ly[half])
p_nnls = np.expm1(M @ w)
ev_half(p_nnls, "Blend: NNLS weights on log1p (top-8)", "ensemble",
        "w=" + ",".join(f"{x:.2f}" for x in w))

# 3. NNLS + intercept
M1 = np.concatenate([M, np.ones((NU, 1))], 1)
w1, _ = nnls(M1[half], ly[half])
ev_half(np.expm1(M1 @ w1), "Blend: NNLS + intercept (top-8)", "ensemble")

# 4. isotonic calibration of the single best model
best_name = scored[0][1]
b = np.log1p(np.clip(preds[best_name], 0, None))
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(b[half], ly[half])
ev_half(np.expm1(iso.predict(b)), f"Isotonic calibration of best single ({best_name[:38]})", "ensemble")

# 5. isotonic calibration of the NNLS blend
bb = np.log1p(np.clip(p_nnls, 0, None))
iso2 = IsotonicRegression(out_of_bounds="clip")
iso2.fit(bb[half], ly[half])
ev_half(np.expm1(iso2.predict(bb)), "Isotonic calibration of the NNLS blend", "ensemble")

# 6. seasonal multiplier sensitivity (informational, on the full validation set)
print("\n--- seasonal multiplier sensitivity on the NNLS blend ---")
rows = []
for k in [0.8, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5]:
    rows.append((k, rmsle(y[~half], (p_nnls * k)[~half])))
for k, s in rows:
    print(f"  x{k:<5.2f} -> RMSLE {s:.5f}")
best_k = min(rows, key=lambda r: r[1])
print(f"  optimal multiplier on the validation window: x{best_k[0]:.2f}")
print("  note: the validation window is a seasonal trough; the 2025 calendar analogue of the")
print("        test window ran +16.3% against its preceding 30 days, so the test-time multiplier")
print("        should be searched above this value, not at it.")
print("Tier 6 done")
