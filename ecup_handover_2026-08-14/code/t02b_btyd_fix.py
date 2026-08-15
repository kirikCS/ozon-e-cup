"""Tier 2b: fixes for the two broken Tier-2 entries + the empirical-Bayes log-block model.

Bug 1: Gamma-Gamma occasionally converged to q <= 1 (infinite population mean) -> exploding predictions.
       Fixed by constraining q > 1 and clipping the conditional expectation.
Bug 2: E[log1p(sum over 30 days)] != 30 * p * E[log1p(daily spend)]. The sum-of-logs decomposition
       does not hold. Replaced by an empirical-Bayes estimator of E[log1p(30-day block sum)],
       which is exactly the quantity RMSLE asks for.
"""
import pickle
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, TRAIN_ANCHORS, H, NU, panel, target,
                    cohort_rule, rmsle, record, Timer, ds, OUT)

A, AH = VAL_ANCHOR, HON_ANCHOR
FIT_ANCHORS = TRAIN_ANCHORS[:4]
G = panel("gmv"); V = panel("visit")
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
BUY = (G > 0)
csB = np.concatenate([np.zeros((NU, 1), np.int64), BUY.astype(np.int64)], 1).cumsum(1)
wG = lambda a, b: csG[:, max(b, 0) + 1] - csG[:, max(a, 0)]
y_val, y_hon = target(A), target(AH)
m_hon = cohort_rule(AH)
SUB = np.random.default_rng(0).choice(NU, 25000, replace=False)

def ev(fn, name, family, notes=""):
    with Timer() as t:
        pv, ph = fn(A), fn(AH)
    return record(name, family, rmsle(y_val, pv), rmsle(y_hon[m_hon], ph[m_hon]), t.dt, notes)

def logcal(fn, name, family, notes=""):
    """calibration grid on pre-computed log-space arrays: no per-cell recomputation"""
    LP = {a: np.log1p(np.clip(fn(a), 0, None)) for a in FIT_ANCHORS}
    LT = {a: np.log1p(target(a)) for a in FIT_ANCHORS}
    best = (1e9, (1.0, 0.0))
    for b in np.linspace(0.2, 1.4, 49):
        for c in np.linspace(-0.5, 2.5, 61):
            s = np.mean([float(np.sqrt(np.mean((LT[a] - np.maximum(b * LP[a] + c, 0)) ** 2)))
                         for a in FIT_ANCHORS])
            if s < best[0]: best = (s, (b, c))
    b, c = best[1]
    f = lambda a: np.expm1(b * np.log1p(np.clip(fn(a), 0, None)) + c)
    return record(name, family, rmsle(y_val, f(A)), rmsle(y_hon[m_hon], f(AH)[m_hon]), None,
                  notes + f" | calib b={b:.2f} c={c:.2f}")

# --------------------------------------------------- fix 1: robust Gamma-Gamma
def rfm(a):
    sub = V[:, :a + 1] > 0
    first = np.argmax(sub, axis=1)
    bsub = BUY[:, :a + 1]
    has = bsub.any(axis=1)
    last_buy = a - np.argmax(bsub[:, ::-1], axis=1)
    n_buy = csB[:, a + 1]
    x = np.maximum(n_buy - 1, 0).astype(np.float64)
    t_x = np.minimum(np.where(has, last_buy - first, 0).astype(np.float64), (a - first).astype(np.float64))
    T = (a - first).astype(np.float64)
    mval = np.where(n_buy > 0, csG[:, a + 1] / np.maximum(n_buy, 1), 0.0)
    return x, t_x, T, n_buy, mval

def gg_nll(par, nb, mval):
    p, q1, g = np.exp(par)
    q = 1.0 + q1                                  # hard constraint q > 1
    ll = (gammaln(p * nb + q) - gammaln(p * nb) - gammaln(q)
          + q * np.log(g) + (p * nb - 1) * np.log(mval) + (p * nb) * np.log(nb)
          - (p * nb + q) * np.log(g + nb * mval))
    v = -ll.sum()
    return v if np.isfinite(v) else 1e18

def gg_fit(nb, mval):
    m = (nb > 0) & (mval > 0)
    if m.sum() > 40000:
        idx = np.flatnonzero(m)[np.random.default_rng(1).choice(int(m.sum()), 40000, replace=False)]
        nb, mval = nb[idx], mval[idx]
    else:
        nb, mval = nb[m], mval[m]
    seed = _WARM["gg"] if _WARM["gg"] is not None else np.log([1., 1., 1.])
    r = minimize(gg_nll, seed, args=(nb, mval), method="Nelder-Mead",
                 options={"maxiter": 3000, "maxfev": 3000, "xatol": 1e-4, "fatol": 1e-2})
    _WARM["gg"] = r.x
    p, q1, g = np.exp(r.x)
    return p, 1.0 + q1, g

def gg_expected(par, nb, mval):
    p, q, g = par
    pop = g * p / (q - 1)
    pop = float(np.clip(pop, 1, 1e4))
    w = p * nb / (p * nb + q - 1)
    out = np.where(nb > 0, w * mval + (1 - w) * pop, pop)
    return np.clip(np.nan_to_num(out), 0, 1e5)

def _unpack(par):
    """a is constrained to > 1: the BG/NBD conditional expectation has a (a-1) denominator
    and is undefined otherwise (this is what produced all-zero predictions before)."""
    r, alpha, bb = np.exp(par[[0, 1, 3]])
    aa = 1.0 + np.exp(par[2])
    return r, alpha, aa, bb

def bgnbd_nll(par, x, t_x, T):
    r, alpha, aa, bb = _unpack(par)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(aa + bb) + gammaln(bb + x) - gammaln(bb) - gammaln(aa + bb + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(aa) - np.log(bb + x - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
    v = -(a1 + a2 + np.logaddexp(a3, a4)).sum()
    return v if np.isfinite(v) else 1e18

from scipy.special import hyp2f1
_WARM = {"bg": None, "gg": None}
def bgnbd_fit(x, t_x, T):
    """warm-started from the previous anchor's solution: the parameters move very little
    between anchors, so one restart from there converges in a fraction of the evaluations"""
    seeds = [[0., 0., 0., 0.], [0.5, 1.0, -0.5, 0.5]] if _WARM["bg"] is None else [_WARM["bg"]]
    best, bp = np.inf, None
    for seed in seeds:
        r = minimize(bgnbd_nll, np.array(seed, dtype=float), args=(x, t_x, T), method="Nelder-Mead",
                     options={"maxiter": 3000, "maxfev": 3000, "xatol": 1e-4, "fatol": 1e-2})
        if r.fun < best: best, bp = r.fun, r.x
    _WARM["bg"] = bp
    return _unpack(bp)

def bgnbd_expected(par, x, t_x, T, t):
    r, alpha, aa, bb = par
    first = (aa + bb + x - 1) / (aa - 1)
    hz = t / (alpha + T + t)
    F = hyp2f1(r + x, bb + x, aa + bb + x - 1, hz)
    second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * F
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.clip(np.nan_to_num(first * second / den), 0, 1e4)

def bgnbd_palive(par, x, t_x, T):
    r, alpha, aa, bb = par
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.nan_to_num(1.0 / (1.0 + ratio))

_C = {}
def bgpar(a):
    if a not in _C:
        x, t_x, T, nb, mv = rfm(a)
        _C[a] = (bgnbd_fit(x[SUB], t_x[SUB], T[SUB]), x, t_x, T, nb, mv)
    return _C[a]

_GG = {}
def ggpar(a):
    if a not in _GG:
        _, _, _, _, nb, mv = bgpar(a)
        _GG[a] = gg_fit(nb, mv)
    return _GG[a]

def bg_gg(a):
    par, x, t_x, T, nb, mv = bgpar(a)
    return bgnbd_expected(par, x, t_x, T, H) * gg_expected(ggpar(a), nb, mv)

print("--- Tier 2b: fixes ---")
for a in [A, AH]:
    par = bgpar(a)[0]
    ex = bgnbd_expected(par, *bgpar(a)[1:4], H)
    print(f"  {ds(a)}: BG/NBD r={par[0]:.4f} alpha={par[1]:.3f} a={par[2]:.4f} b={par[3]:.4f} "
          f"| E[X(30)] mean={ex.mean():.3f} zeros={float((ex == 0).mean()):.3f}")
    print(f"  {ds(a)}: Gamma-Gamma p={ggpar(a)[0]:.4f} q={ggpar(a)[1]:.4f} g={ggpar(a)[2]:.3f}")
ev(bg_gg, "BG/NBD x Gamma-Gamma [fixed]", "btyd", "q constrained > 1")
logcal(bg_gg, "BG/NBD x Gamma-Gamma [fixed] + log-calibration", "btyd")

# ------------------------------------- fix 2: empirical-Bayes over 30-day blocks
def blocks_log(a, K):
    return np.stack([np.log1p(wG(a - 30 * i - 29, a - 30 * i)) for i in range(K)], 1)

def eb_logblocks(a, K, kappa, decay=1.0):
    L = blocks_log(a, K)
    w = decay ** np.arange(K)
    pop = (L * w).sum() / (w.sum() * NU)
    num = (L * w).sum(1) + kappa * pop
    den = w.sum() + kappa
    return np.expm1(num / den)

grid = [(K, k, d) for K in [4, 6, 8, 10, 12] for k in [0.0, 0.5, 1.0, 2.0, 4.0] for d in [1.0, 0.9, 0.8, 0.7]]
best = (1e9, None)
for K, k, d in grid:
    s = np.mean([rmsle(target(a), eb_logblocks(a, K, k, d)) for a in FIT_ANCHORS])
    if s < best[0]: best = (s, (K, k, d))
K, kap, dec = best[1]
print(f"  EB log-blocks tuned: K={K} kappa={kap} decay={dec}")
ev(lambda a: eb_logblocks(a, K, kap, dec),
   f"EB shrunk mean of log1p 30d blocks (K={K}, kappa={kap}, decay={dec})", "btyd",
   "estimates E[log1p(y)] directly; params fitted on train")
logcal(lambda a: eb_logblocks(a, K, kap, dec),
       "EB log-blocks + log-calibration", "btyd")

# blocks of other lengths
for W in [14, 60, 90]:
    def ebw(a, W=W):
        Kw = min(10, (a + 1) // W)
        L = np.stack([np.log1p(wG(a - W * i - W + 1, a - W * i) * H / W) for i in range(Kw)], 1)
        pop = L.mean()
        return np.expm1((L.sum(1) + kap * pop) / (Kw + kap))
    logcal(ebw, f"EB log-blocks of {W}d rescaled to 30d + calibration", "btyd")

# --------------------------------------------------- export BTYD features
feats = {}
want = sorted(set(TRAIN_ANCHORS + [A, AH] + list(range(160, 272, 14))))
for a in want:
    par, x, t_x, T, nb, mv = bgpar(a)
    p_daily = (csB[:, a + 1] + 1.0) / (a + 1 + 20.0)
    feats[a] = np.stack([
        bgnbd_expected(par, x, t_x, T, H),
        bgnbd_palive(par, x, t_x, T),
        gg_expected(ggpar(a), nb, mv),
        p_daily,
        np.log1p(eb_logblocks(a, K, kap, dec)),
    ], 1).astype(np.float32)
with open(OUT / "btyd_feats.pkl", "wb") as f:
    pickle.dump(feats, f)
print(f"  BTYD features saved for {len(feats)} anchors, 5 columns")
print("Tier 2b done")
