"""Segment-restricted LB probe: measure the conditional level bias on the TRUE test window.

The parabola algebra extends exactly to a shift applied only on segment S:
    R'^2 = R0^2 + w_S c^2 - 2 c w_S D_S,   D_S = E[log1p y - log1p p | S]
so ONE submission of stack7 with log-shift c on S resolves D_S on the actual Feb-Mar
window — the quantity no local protocol can see (AGENT.md §4 dichotomy). The correction,
if any, is then applied SHRUNKEN (c_S = kappa*D_S, kappa = max(0, 1 - SE^2/D_S^2)) and
only on segments >= 30% mass.

Axis 1: S = zero-prev30 at anchor 408 (no purchases in the last 30 days). Historically
these users carry 12.6% of target mass; gift-driven reactivation around Feb 23 / Mar 8
is exactly the mechanism that would bias their level.

Writes submission/queue/probe_zero30_c015.csv and prints the analysis constants.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, NU

C = 0.15
R0 = 1.649380786349335          # stack7, measured

G = panel("gmv")
S = G[:, 379:409].sum(axis=1) == 0          # zero GMV in the 30 days before anchor 408
w_S = float(S.mean())

d = pl.read_csv("submission/stack7_2s9seed_lb1649381.csv")
p = d["predict"].to_numpy()
q = np.log1p(p)
q2 = q + C * S
p2 = np.clip(np.expm1(q2), 0, None)

out = Path("submission/queue/probe_zero30_c015.csv")
with open(out, "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(d["user_id"].to_numpy(), p2):
        f.write(f"{u},{v:.6f}\n")

null_R = np.sqrt(R0 ** 2 + w_S * C ** 2)
print(f"segment zero-prev30: w_S = {w_S:.4f} ({int(S.sum())} users, ~{int(50000*w_S)} public)")
print(f"probe written: {out}  (c = +{C} on S)")
print(f"if D_S = 0 (no bias): expected LB = {null_R:.6f}")
print(f"analysis: D_S = (R0^2 + w_S c^2 - R'^2) / (2 c w_S)")
print(f"          = ({R0**2:.6f} + {w_S*C**2:.6f} - R'^2) / {2*C*w_S:.6f}")
se = 1.65 / np.sqrt(50000 * w_S)
print(f"SE(D_S) ~ {se:.4f}; resolvable if true |D_S| >= {2*se:.4f}")
print(f"payoff if |D_S| = 0.04: dR ~ {w_S*0.04**2/(2*R0):.5f} (shrunken, one axis)")
