"""BG/NBD + Gamma-Gamma + empirical-Bayes helpers, extracted from t02b so they can be reused
for anchors that were not covered by the original feature export."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

_WARM = {"bg": None, "gg": None}

def _unpack(par):
    """a is constrained to > 1: the BG/NBD conditional expectation divides by (a-1)."""
    r, alpha, bb = np.exp(par[[0, 1, 3]])
    return r, alpha, 1.0 + np.exp(par[2]), bb

def bgnbd_nll(par, x, t_x, T):
    r, alpha, aa, bb = _unpack(par)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(aa + bb) + gammaln(bb + x) - gammaln(bb) - gammaln(aa + bb + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(aa) - np.log(bb + x - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
    v = -(a1 + a2 + np.logaddexp(a3, a4)).sum()
    return v if np.isfinite(v) else 1e18

def bgnbd_fit(x, t_x, T):
    seeds = [[0., 0., 0., 0.], [0.5, 1.0, -0.5, 0.5]] if _WARM["bg"] is None else [_WARM["bg"]]
    best, bp = np.inf, None
    for s in seeds:
        r = minimize(bgnbd_nll, np.array(s, dtype=float), args=(x, t_x, T), method="Nelder-Mead",
                     options={"maxiter": 3000, "maxfev": 3000, "xatol": 1e-4, "fatol": 1e-2})
        if r.fun < best: best, bp = r.fun, r.x
    _WARM["bg"] = bp
    return _unpack(bp)

def bgnbd_expected(par, x, t_x, T, t):
    r, alpha, aa, bb = par
    first = (aa + bb + x - 1) / (aa - 1)
    hz = t / (alpha + T + t)
    second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, bb + x, aa + bb + x - 1, hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.clip(np.nan_to_num(first * second / den), 0, 1e4)

def bgnbd_palive(par, x, t_x, T):
    r, alpha, aa, bb = par
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.nan_to_num(1.0 / (1.0 + ratio))

def gg_nll(par, nb, mval):
    p, q1, g = np.exp(par)
    q = 1.0 + q1
    ll = (gammaln(p * nb + q) - gammaln(p * nb) - gammaln(q) + q * np.log(g)
          + (p * nb - 1) * np.log(mval) + (p * nb) * np.log(nb) - (p * nb + q) * np.log(g + nb * mval))
    v = -ll.sum()
    return v if np.isfinite(v) else 1e18

def gg_fit(nb, mval, rng_seed=1):
    m = (nb > 0) & (mval > 0)
    if m.sum() > 40000:
        idx = np.flatnonzero(m)[np.random.default_rng(rng_seed).choice(int(m.sum()), 40000, replace=False)]
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
    pop = float(np.clip(g * p / (q - 1), 1, 1e4))
    w = p * nb / (p * nb + q - 1)
    return np.clip(np.nan_to_num(np.where(nb > 0, w * mval + (1 - w) * pop, pop)), 0, 1e5)
