"""Sanity check that the locally rebuilt panels reproduce the published numbers.

Every figure below is quoted in text.md / the EDA report, so any mismatch means the panels
are wrong, not that the report is.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (VAL_ANCHOR, HON_ANCHOR, H, NU, panel, target, cohort_rule,
                    rmsle, ds, CACHE)

print(f"cache: {CACHE}")
A, AH = VAL_ANCHOR, HON_ANCHOR
G = panel("gmv")
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
wG = lambda a, b: csG[:, max(b, 0) + 1] - csG[:, max(a, 0)]

y = target(A)
checks = []

def chk(label, got, want, tol):
    ok = abs(got - want) <= tol
    checks.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {label:<46s} {got:>12.5f}   expected {want}")

print(f"\nvalidation anchor {ds(A)} -> [{ds(A+1)}, {ds(A+H)}]")
chk("RMSLE prev30 (sample_submit baseline)", rmsle(y, wG(A - 29, A)), 2.19506, 5e-5)
chk("RMSLE prev90/3", rmsle(y, wG(A - 89, A) / 3), 2.09355, 5e-5)
chk("RMSLE prev60/2", rmsle(y, wG(A - 59, A) / 2), 2.11140, 5e-5)
chk("RMSLE const 0", rmsle(y, np.zeros(NU)), 3.20364, 5e-5)

print("\ntarget structure at the validation anchor")
chk("share of zeros", float((y == 0).mean()), 0.4593, 5e-4)
chk("mean target", float(y.mean()), 84.0, 0.1)
chk("max target", float(y.max()), 53747.0, 1.0)

print("\nselection rule (the key EDA finding)")
V = panel("visit")
for k in range(5):
    e = 408 - 30 * k
    s = e - 29
    share = float((V[:, s:e + 1].sum(1) > 0).mean())
    want = [1.0, 1.0, 1.0, 0.9326, 0.9017][k]
    chk(f"block -{k}  {ds(s)}..{ds(e)}", share, want, 5e-4)

print("\nhonest anchor")
m = cohort_rule(AH)
chk("cohort size at 2025-10-16", float(m.sum()), 197379, 0)
chk("RMSLE prev30 on the honest cohort", rmsle(target(AH)[m], wG(AH - 29, AH)[m]), 2.23620, 5e-5)

print("\npanel integrity")
chk("panel fill rate", float((V > 0).mean()), 0.2996, 5e-4)
chk("rows in the panel", float((V > 0).sum()), 30631006, 0)

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
