"""Does a new feature subdivide the oracle cells, i.e. move the CEILING — model-free?

oracle_floor.py showed the irreducible noise: users bucketed by the 5 strongest features,
exact in-cell mean of log1p(target), corrected for in-sample fitting -> sigma = 1.6456.
A new feature is worth pursuing ONLY if adding it to the grid lowers that conditional
spread. If the oracle does not move, no model of any kind can profit from the feature at
this resolution — the direction is closed by theory, not by one model family.

Estimator (cleaner than the original's global (1-1/n) correction): random split-half.
Cell means are fitted on half A and evaluated on half B, so the estimate is out-of-sample
by construction; the exact per-cell correction E[(y_B - meanA)^2] = sigma^2 (1 + 1/nA)
is inverted cell by cell. Both directions (A->B, B->A) are averaged. Cells that lack
members in either half drop out — coverage is reported, and comparisons are only valid
at similar coverage (the 6th dimension uses few bins for exactly that reason).

Run at the VAL anchor like oracle_floor.py: this is a diagnostic bound on the data, not a
model selection. A feature that moves the oracle still has to win on the SELECT screen;
a feature that does not move it is closed outright.

Usage:  python -u code/fx_oracle.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import VAL_ANCHOR, NU, target, build_features
import fx

A = VAL_ANCHOR
BASE5 = ["buyday_all", "gmv_90", "buyday_30", "rec_buy", "gmv_30"]
NB_BASE = 20          # quantile bins for the base 5 (as in oracle_floor.py: 21 buckets)
NB_NEW = 4            # coarse bins for the candidate 6th dimension, keeps coverage high


def quant(v, nb):
    qs = np.percentile(v, np.linspace(100 / nb, 100 - 100 / nb, nb - 1))
    return np.searchsorted(qs, v).astype(np.int64)


def sigma_split(cols_q, ly, rng):
    """cols_q: list of int arrays; returns (sigma_hat, coverage)"""
    key = cols_q[0].copy()
    for c in cols_q[1:]:
        key = key * (c.max() + 1) + c
    _, key = np.unique(key, return_inverse=True)
    half = rng.random(NU) < 0.5
    out = []
    cov = []
    for mA in (half, ~half):
        mB = ~mA
        nc = key.max() + 1
        nA = np.bincount(key[mA], minlength=nc)
        sA = np.bincount(key[mA], weights=ly[mA], minlength=nc)
        ok = nA > 0
        meanA = np.divide(sA, nA, out=np.zeros_like(sA), where=ok)
        useB = ok[key] & mB
        r2 = (ly[useB] - meanA[key[useB]]) ** 2
        w = 1.0 + 1.0 / nA[key[useB]]
        out.append(r2.sum() / w.sum())
        cov.append(useB.sum() / mB.sum())
    return float(np.sqrt(np.mean(out))), float(np.mean(cov))


def main():
    X, names = build_features([A])
    Xa = X[A]
    ly = np.log1p(target(A))
    base_q = [quant(Xa[:, names.index(n)].astype(np.float64), NB_BASE) for n in BASE5]

    # candidate 6th dimensions from the extension groups
    import os
    if os.environ.get("FX_ORACLE_GROUP") == "holiday":
        cand_specs = [("holiday", ["hol_f23_lift", "hol_m8_lift", "hol_gift_lift",
                                   "hol_yago_gmv", "hol_yago_bdays", "hol_yago_share",
                                   "hol_ny_lift"])]
    else:
        cand_specs = [("gap", ["b_gap_cv", "b_gap_std", "b_gap_trend", "b_open_over_max",
                               "b_gap_last", "v_gap_cv"]),
                      ("dow", ["dow_wknd_gmv_all", "dow_ent_gmv_all", "dow_top_gmv_all",
                               "dow_ent_visit_all"]),
                      ("chan", ["rec_sord", "day_srch_share_all", "ord_srch_share_all",
                                "sday_all", "catday_all"]),
                      ("funnel", ["s2o_all", "sc2o_all", "cc2o_all", "cart_unconv_30"])]

    # Adding ANY 6th dimension biases sigma down through cell splitting and re-weighting
    # (a pure-noise dimension "gained" -0.042), and a GLOBAL shuffle over-splits relative
    # to the true column (real features correlate with the base grid, so their joint cells
    # are fewer) — both controls mismatch the artefact. The exact null is a WITHIN-CELL
    # permutation: inside every base cell the candidate's bins are shuffled among members,
    # which preserves the joint cell structure bit for bit and destroys only the residual
    # association with the target. signal = sigma(true) - sigma(within-shuffled);
    # negative beyond the spread = the feature genuinely lowers the ceiling.
    sig0 = np.array([sigma_split(base_q, ly, np.random.default_rng(100 + r))[0]
                     for r in range(8)])
    print(f"base grid ({NB_BASE} bins x 5): sigma {sig0.mean():.5f} +- {sig0.std():.5f}")

    key0 = base_q[0].copy()
    for c in base_q[1:]:
        key0 = key0 * (c.max() + 1) + c

    def within_shuffle(fq, rng):
        i0 = np.lexsort((rng.random(NU), key0))
        i1 = np.lexsort((rng.random(NU), key0))
        out = np.empty_like(fq)
        out[i0] = fq[i1]
        return out

    NPERM = 4
    for spec, feats in cand_specs:
        fn, FX = fx.build_group(spec, [A])
        for f in feats:
            v = FX[A][:, fn.index(f"{spec}:{f}")].astype(np.float64)
            fq = quant(v, NB_NEW)
            ss = np.array([sigma_split(base_q + [fq], ly, np.random.default_rng(100 + r))[0]
                           for r in range(8)])
            _, cov = sigma_split(base_q + [fq], ly, np.random.default_rng(100))
            perm = []
            for pi in range(NPERM):
                pq = within_shuffle(fq, np.random.default_rng(500 + pi))
                perm.append(np.mean([sigma_split(base_q + [pq], ly,
                                                 np.random.default_rng(100 + r))[0]
                                     for r in range(8)]))
            perm = np.array(perm)
            d = ss.mean() - perm.mean()
            print(f"+ {spec}:{f:<22s} true {ss.mean():.5f}  null {perm.mean():.5f} "
                  f"+- {perm.std():.5f}  signal {d:+.5f}  cov {cov:.3f}")


if __name__ == "__main__":
    main()
