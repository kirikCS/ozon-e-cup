"""ElasticNet / L1 as a base model on the 296 features.

At the stacking layer regularisation had nothing to do — C and k are known exactly from the
leaderboard, so there is no estimation noise to shrink. Here it is the opposite: 296 correlated
columns fitted on a million rows, which is the regime L1 was designed for.

The 46% zero mass is the reason to expect the two-stage form to beat the single regression: a linear
model cannot bend to a point mass at zero, but it can model log-odds of purchase and conditional
log1p spend separately, and the exact identity E[log1p y] = P(y>0) * E[log1p y | y>0] then puts them
back together with no correction term.

Four variations:
    en_plain     ElasticNet on log1p y, grid over alpha and l1_ratio
    en_lasso     pure L1, to see how few features the linear model actually needs
    en_two       L1 logistic for P(y>0) times ElasticNet fitted on positives only
    en_ple       ElasticNet on a piecewise-linear-encoded basis: each top feature split into
                 quantile bins, so the linear model can express monotone-but-curved effects
"""
import sys
import time

import numpy as np
from sklearn.linear_model import ElasticNet, SGDClassifier

sys.path.insert(0, "models")
from zoo_common import load, emit, TR_PROBE, TR_FINAL, FIT, PRED
from common import rmsle, ds

X, Y, NAMES, TOP = load()
RNG = np.random.default_rng(0)
SUB = 500_000                       # rows for the grid search; the final fit uses everything

def stackXY(anchors):
    return (np.concatenate([X[a] for a in anchors], 0).astype(np.float64),
            np.concatenate([Y[a] for a in anchors]))

# ---------------------------------------------------------------- PLE basis
NB = 8
def ple_edges(Xtr, cols):
    return {c: np.quantile(Xtr[:, c], np.linspace(0, 1, NB + 1)[1:-1]) for c in cols}

def ple_apply(Xm, edges, cols):
    """each selected column becomes NB ramps: 0 below the bin, linear inside, 1 above.
    piecewise-linear rather than one-hot, so the encoding stays monotone and needs no interactions"""
    out = [Xm]
    for c in cols:
        e = np.concatenate([[Xm[:, c].min() - 1e-6], edges[c], [Xm[:, c].max() + 1e-6]])
        v = Xm[:, c:c + 1]
        lo, hi = e[:-1][None, :], e[1:][None, :]
        out.append(np.clip((v - lo) / np.maximum(hi - lo, 1e-9), 0, 1))
    return np.concatenate(out, 1)

def run(name, fname, fitter, use_ple=False):
    t0 = time.time()
    for anchors, ev, final in ((TR_PROBE, FIT, False), (TR_FINAL, PRED, True)):
        Xtr, ytr = stackXY(anchors)
        Xev = X[ev].astype(np.float64)
        if use_ple:
            cols = list(TOP[:40])
            ed = ple_edges(Xtr, cols)
            Xtr, Xev = ple_apply(Xtr, ed, cols), ple_apply(Xev, ed, cols)
        p = fitter(Xtr, ytr, Xev)
        del Xtr
        if final:
            emit(np.clip(p, 0, 13), fname)
        else:
            print(f"  {name}: probe {rmsle(Y[ev], np.expm1(np.clip(p,0,13))):.5f}  "
                  f"({Xev.shape[1]} колонок, {time.time()-t0:.0f}s)")

def grid_en(Xtr, ytr, Xev, ratios=(0.1, 0.5, 0.9, 1.0), alphas=(1e-5, 1e-4, 1e-3, 1e-2)):
    """pick alpha / l1_ratio on a held-out slice of the training anchors, then refit on all of it"""
    n = len(ytr)
    idx = RNG.permutation(n)
    tr, va = idx[:min(SUB, int(0.8 * n))], idx[int(0.8 * n):int(0.8 * n) + 150_000]
    ly = np.log1p(ytr)
    best = None
    for r in ratios:
        for al in alphas:
            m = ElasticNet(alpha=al, l1_ratio=r, max_iter=3000, tol=1e-4,
                           selection="random", random_state=0).fit(Xtr[tr], ly[tr])
            s = float(np.sqrt(np.mean((np.clip(m.predict(Xtr[va]), 0, 13) - ly[va]) ** 2)))
            nz = int((np.abs(m.coef_) > 1e-9).sum())
            if best is None or s < best[0]:
                best = (s, al, r, nz)
    s, al, r, nz = best
    print(f"    выбрано alpha={al}, l1_ratio={r} (hold-out {s:.5f}, ненулевых {nz}/{Xtr.shape[1]})")
    m = ElasticNet(alpha=al, l1_ratio=r, max_iter=3000, tol=1e-4,
                   selection="random", random_state=0).fit(Xtr[:SUB * 2], ly[:SUB * 2])
    return m.predict(Xev)

def lasso_only(Xtr, ytr, Xev):
    return grid_en(Xtr, ytr, Xev, ratios=(1.0,), alphas=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3))

def two_stage(Xtr, ytr, Xev):
    """P(y>0) from an L1-penalised logistic, conditional log1p mean from ElasticNet on positives"""
    pos = ytr > 0
    clf = SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=0.3, alpha=1e-6,
                        max_iter=15, tol=1e-4, random_state=0)
    sl = slice(0, min(len(ytr), SUB * 2))
    clf.fit(Xtr[sl], pos[sl])
    p1 = clf.predict_proba(Xev)[:, 1]
    ip = np.where(pos)[0][:SUB]
    m = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=3000, tol=1e-4,
                   selection="random", random_state=0).fit(Xtr[ip], np.log1p(ytr[ip]))
    print(f"    ненулевых в логистике {int((np.abs(clf.coef_)>1e-9).sum())}, "
          f"в регрессии {int((np.abs(m.coef_)>1e-9).sum())}")
    return p1 * np.clip(m.predict(Xev), 0, 13)

print("=== ElasticNet / L1 как базовая модель ===")
run("ElasticNet", "en_plain", grid_en)
run("Lasso (чистый L1)", "en_lasso", lasso_only)
run("Two-stage L1", "en_two", two_stage)
run("ElasticNet + PLE", "t26_elastic_base", grid_en, use_ple=True)
print("\ndone")
