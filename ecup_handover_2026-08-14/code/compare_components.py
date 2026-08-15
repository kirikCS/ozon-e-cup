"""Compare freshly retrained submissions against the shipped components.

The shipped `two_stage_btyd_shift.csv` is sub2 with a +0.105 log-shift already applied
(that file is what was scored as sub4), so the comparison has to add the same shift to the
fresh sub2 before diffing. Everything is compared in log1p space, which is the space the
metric lives in: a 1e-3 gap there is a 1e-3 gap in RMSLE terms, whatever the raw magnitude.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

NEW = Path(sys.argv[1] if len(sys.argv) > 1 else "submissions")
OLD = Path(sys.argv[2] if len(sys.argv) > 2 else "submission/components")

# fresh file, shipped file, log-shift to apply to the fresh one
PAIRS = [
    ("sub1_lgbm_dart.csv", "lgbm_dart.csv", 0.0),
    ("sub2_two_stage_btyd.csv", "two_stage_btyd_shift.csv", 0.105),
    ("sub3_nnls_isotonic_blend.csv", "nnls_isotonic_blend.csv", 0.0),
    ("sub8_catboost_rmse_log1p_shifted.csv", "catboost_shifted.csv", 0.0),
]

print(f"{'component':<34} {'max |dlog1p|':>13} {'mean |d|':>11} {'corr':>8}  verdict")
for new, old, shift in PAIRS:
    pn, po = NEW / new, OLD / old
    if not pn.exists():
        print(f"{old:<34} {'—':>13} {'—':>11} {'—':>8}  not retrained yet ({new})")
        continue
    a, b = pl.read_csv(pn), pl.read_csv(po)
    assert (a["user_id"].to_numpy() == b["user_id"].to_numpy()).all(), f"user_id order differs: {new}"
    la = np.log1p(a["predict"].to_numpy()) + shift
    lb = np.log1p(b["predict"].to_numpy())
    d = np.abs(la - lb)
    c = float(np.corrcoef(la, lb)[0, 1])
    verdict = ("identical" if d.max() < 1e-4 else
               "reproduced" if d.max() < 5e-3 else
               "close" if c > 0.999 else "DIFFERS")
    print(f"{old:<34} {d.max():>13.2e} {d.mean():>11.2e} {c:>8.6f}  {verdict}")
