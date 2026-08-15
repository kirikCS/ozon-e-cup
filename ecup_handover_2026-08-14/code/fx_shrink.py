"""Per-segment affine shrinkage in log space — the one transform a linear stack cannot express.

Users differ in predictability (history length drives estimation noise); James-Stein logic
says noisier predictions should shrink harder toward the mean. Per-segment (buyday_all
deciles) affine maps ly' = b_s * p + a_s are fitted on the fully-observable anchor 348 and
applied FROZEN at VAL (378) and HON (288). Controls: global affine (1 segment) — the gain
must exceed it; and sign agreement between VAL and HON (the isotonic/gate lesson).

Uses the cached 9-seed two-stage ensembles per protocol. Pure vector algebra, no fits.

Usage:  python -u code/fx_shrink.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, target, rmsle, cohort_rule, OUT, NU

SEEDS = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
NSEG = 10


def buyday_all(a):
    return (panel("gmv")[:, :a + 1] > 0).sum(axis=1)


def ens(model, proto):
    tag = "" if proto == "SELECT" else f"_{proto}"
    return np.mean([np.load(OUT / f"fx_se_{model}{tag}_s{s}.npy") for s in SEEDS], 0)


# fit segment maps at anchor 348 (observable history)
FIT_A = 348
p_fit = ens("2s", "SELECT")
y_fit = np.log1p(target(FIT_A))
seg_fit = buyday_all(FIT_A)
qs = np.percentile(seg_fit, np.arange(10, 100, 10))


def seg_of(v):
    return np.searchsorted(qs, v)


maps = {}
for s in range(NSEG):
    m = seg_of(seg_fit) == s
    A = np.stack([p_fit[m], np.ones(m.sum())], 1)
    b, a = np.linalg.lstsq(A, y_fit[m], rcond=None)[0]
    maps[s] = (b, a)
    print(f"  seg {s}: n {m.sum():>6}  b {b:+.4f}  a {a:+.4f}")

A = np.stack([p_fit, np.ones_like(p_fit)], 1)
bg, ag = np.linalg.lstsq(A, y_fit, rcond=None)[0]
print(f"  global: b {bg:+.4f}  a {ag:+.4f}")


def apply_maps(p, seg):
    out = p.copy()
    s = seg_of(seg)
    for k in range(NSEG):
        m = s == k
        b, a = maps[k]
        out[m] = b * p[m] + a
    return out


for proto, ev, use_mask in [("VAL", 378, False), ("HON", 288, True)]:
    p = ens("2s", proto)
    y = target(ev)
    mask = cohort_rule(ev) if use_mask else None
    seg = buyday_all(ev)

    def sc(v):
        vv = v[mask] if mask is not None else v
        yy = y[mask] if mask is not None else y
        return rmsle(yy, np.expm1(np.clip(vv, 0, None)))

    base = sc(p)
    glob = sc(bg * p + ag)
    segm = sc(apply_maps(p, seg))
    print(f"{proto}: base {base:.5f}  global-affine {glob:.5f} ({glob-base:+.5f})  "
          f"per-segment {segm:.5f} ({segm-base:+.5f})  beyond-global {segm-glob:+.5f}")
