"""Tier 2: BTYD / probabilistic CLV. BG/NBD, MBG/NBD, Gamma-Gamma, empirical-Bayes Beta-Binomial.
All parameters are fitted on the history available at each anchor, never on the target.
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, betaln, hyp2f1, logsumexp

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, TRAIN_ANCHORS, H, NU, panel, target,
                    cohort_rule, rmsle, record, Timer, ds)

A, AH = VAL_ANCHOR, HON_ANCHOR
FIT_ANCHORS = TRAIN_ANCHORS[:4]
G = panel("gmv"); V = panel("visit")
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
BUY = (G > 0)
csB = np.concatenate([np.zeros((NU, 1), np.int64), BUY.astype(np.int64)], 1).cumsum(1)
y_val, y_hon = target(A), target(AH)
m_hon = cohort_rule(AH)

def rfm(a):
    """customer 'birth' = first day with a record; frequency = repeat buy-days."""
    sub = V[:, :a + 1] > 0
    first = np.argmax(sub, axis=1)
    bsub = BUY[:, :a + 1]
    has = bsub.any(axis=1)
    last_buy = a - np.argmax(bsub[:, ::-1], axis=1)
    n_buy = csB[:, a + 1]
    x = np.maximum(n_buy - 1, 0).astype(np.float64)          # repeat transactions
    t_x = np.where(has, last_buy - first, 0).astype(np.float64)
    T = (a - first).astype(np.float64)
    t_x = np.minimum(t_x, T)
    gmv = csG[:, a + 1]
    mval = np.where(n_buy > 0, gmv / np.maximum(n_buy, 1), 0.0)
    return x, t_x, T, n_buy, mval

# ------------------------------------------------------------------ BG/NBD
def bgnbd_nll(par, x, t_x, T):
    r, alpha, aa, bb = np.exp(par)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(aa + bb) + gammaln(bb + x) - gammaln(bb) - gammaln(aa + bb + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(aa) - np.log(bb + x - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
    ll = a1 + a2 + np.logaddexp(a3, a4)
    return -ll.sum()

def bgnbd_fit(x, t_x, T, seed=(1.0, 1.0, 1.0, 1.0)):
    res = minimize(bgnbd_nll, np.log(seed), args=(x, t_x, T), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-2})
    return np.exp(res.x)

def bgnbd_expected(par, x, t_x, T, t):
    r, alpha, aa, bb = par
    first = (aa + bb + x - 1) / (aa - 1)
    hz = t / (alpha + T + t)
    F = hyp2f1(r + x, bb + x, aa + bb + x - 1, hz)
    second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * F
    num = first * second
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    out = num / den
    return np.nan_to_num(np.clip(out, 0, 1e4))

def bgnbd_palive(par, x, t_x, T):
    r, alpha, aa, bb = par
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(x > 0, (aa / (bb + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return np.nan_to_num(1.0 / (1.0 + ratio))

# ------------------------------------------------------------------ MBG/NBD
def mbgnbd_nll(par, x, t_x, T):
    r, alpha, aa, bb = np.exp(par)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    p1 = betaln(aa, bb + x + 1) - betaln(aa, bb) - (r + x) * np.log(alpha + T)
    p2 = betaln(aa + 1, bb + x) - betaln(aa, bb) - (r + x) * np.log(alpha + t_x)
    ll = a1 + np.logaddexp(p1, p2)
    return -ll.sum()

def mbgnbd_fit(x, t_x, T):
    res = minimize(mbgnbd_nll, np.log([1., 1., 1., 1.]), args=(x, t_x, T), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-2})
    return np.exp(res.x)

def mbgnbd_expected(par, x, t_x, T, t):
    """same conditional expectation machinery, with the MBG dropout convention"""
    r, alpha, aa, bb = par
    first = (aa + bb + x) / (aa - 1)
    hz = t / (alpha + T + t)
    F = hyp2f1(r + x, bb + x + 1, aa + bb + x, hz)
    second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * F
    num = first * second
    den = 1 + (aa / (bb + x)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return np.nan_to_num(np.clip(num / den, 0, 1e4))

# ------------------------------------------------------------------ Gamma-Gamma
def gg_nll(par, x, mval):
    p, q, g = np.exp(par)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
          + q * np.log(g) + (p * x - 1) * np.log(mval) + (p * x) * np.log(x)
          - (p * x + q) * np.log(g + x * mval))
    return -ll.sum()

def gg_fit(x, mval):
    m = (x > 0) & (mval > 0)
    res = minimize(gg_nll, np.log([1., 1., 1.]), args=(x[m], mval[m]), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-2})
    return np.exp(res.x)

def gg_expected(par, x, mval):
    p, q, g = par
    pop = g * p / (q - 1) if q > 1 else float(np.nanmean(mval[mval > 0]))
    # conditional expectation E[M | x, m_x]: shrink the observed mean towards the population mean
    out = np.where(x > 0, ((q - 1) / (p * x + q - 1)) * pop + (p * x / (p * x + q - 1)) * mval, pop)
    return np.nan_to_num(np.clip(out, 0, 1e6))

# ---------------------------------------------- empirical-Bayes Beta-Binomial
def eb_betabin(a, prior_days=None):
    """per-user daily buy probability with a Beta prior fitted by moments on the population"""
    n = np.full(NU, a + 1.0)
    k = csB[:, a + 1].astype(np.float64)
    p = k / n
    mu, var = p.mean(), p.var()
    s = max(mu * (1 - mu) / max(var, 1e-9) - 1, 1e-3)
    a0, b0 = mu * s, (1 - mu) * s
    return (k + a0) / (n + a0 + b0), (a0, b0)

def eb_lognormal_spend(a, shrink=None):
    """per-user mean log1p(daily gmv | buy), shrunk to the population mean"""
    Gs = G[:, :a + 1]
    B = BUY[:, :a + 1]
    s = np.where(B, np.log1p(Gs), 0).sum(1)
    k = B.sum(1).astype(np.float64)
    pop = s.sum() / max(k.sum(), 1)
    kap = 5.0 if shrink is None else shrink
    return (s + kap * pop) / (k + kap), pop

def logcal(pred_fn, name, family, notes=""):
    """fit a 2-parameter log-space calibration on FIT_ANCHORS, then evaluate"""
    best = (1e9, None)
    P = {a: pred_fn(a) for a in FIT_ANCHORS}
    for b in np.linspace(0.2, 1.4, 49):
        for c in np.linspace(-0.5, 2.5, 61):
            sc = np.mean([rmsle(target(a), np.expm1(b * np.log1p(P[a]) + c)) for a in FIT_ANCHORS])
            if sc < best[0]: best = (sc, (b, c))
    b, c = best[1]
    pv = np.expm1(b * np.log1p(pred_fn(A)) + c)
    ph = np.expm1(b * np.log1p(pred_fn(AH)) + c)
    return record(name, family, rmsle(y_val, pv), rmsle(y_hon[m_hon], ph[m_hon]), None,
                  notes + f" | calib b={b:.2f} c={c:.2f}")

def ev(pred_fn, name, family, notes=""):
    with Timer() as t:
        pv, ph = pred_fn(A), pred_fn(AH)
    return record(name, family, rmsle(y_val, pv), rmsle(y_hon[m_hon], ph[m_hon]), t.dt, notes)

print("--- Tier 2: BTYD / probabilistic CLV ---")
SUB = np.random.default_rng(0).choice(NU, 60000, replace=False)   # fit params on a subsample

CACHE = {}
def bg(a):
    if ("bg", a) not in CACHE:
        x, t_x, T, nb, mv = rfm(a)
        par = bgnbd_fit(x[SUB], t_x[SUB], T[SUB])
        CACHE[("bg", a)] = (par, x, t_x, T, nb, mv)
    return CACHE[("bg", a)]

par_v = bg(A)[0]
print(f"  BG/NBD params at {ds(A)}: r={par_v[0]:.4f} alpha={par_v[1]:.3f} a={par_v[2]:.4f} b={par_v[3]:.4f}")

def bg_trans(a):
    par, x, t_x, T, nb, mv = bg(a)
    return bgnbd_expected(par, x, t_x, T, H)

def bg_clv(a):
    par, x, t_x, T, nb, mv = bg(a)
    ex = bgnbd_expected(par, x, t_x, T, H)
    gpar = gg_fit(x[SUB], mv[SUB])
    em = gg_expected(gpar, x, mv)
    return ex * em

def bg_clv_emp(a):
    par, x, t_x, T, nb, mv = bg(a)
    ex = bgnbd_expected(par, x, t_x, T, H)
    pop = mv[nb > 0].mean()
    em = np.where(nb > 0, mv, pop)
    return ex * em

def mbg_clv(a):
    x, t_x, T, nb, mv = rfm(a)
    par = mbgnbd_fit(x[SUB], t_x[SUB], T[SUB])
    ex = mbgnbd_expected(par, x, t_x, T, H)
    pop = mv[nb > 0].mean()
    return ex * np.where(nb > 0, mv, pop)

def eb_clv(a):
    p, _ = eb_betabin(a)
    mlog, pop = eb_lognormal_spend(a)
    # E[log1p(30-day sum)] approximated via expected buy-days x mean log-spend, then exponentiated
    return np.expm1(H * p * mlog)

def eb_mean(a):
    p, _ = eb_betabin(a)
    Gs = G[:, :a + 1]; B = BUY[:, :a + 1]
    s = Gs.sum(1); k = B.sum(1).astype(np.float64)
    pop = s.sum() / max(k.sum(), 1)
    m = (s + 5.0 * pop) / (k + 5.0)
    return H * p * m

ev(bg_trans, "BG/NBD expected transactions (raw count)", "btyd", "no monetary component")
ev(bg_clv, "BG/NBD x Gamma-Gamma", "btyd")
ev(bg_clv_emp, "BG/NBD x empirical mean spend", "btyd")
ev(mbg_clv, "MBG/NBD x empirical mean spend", "btyd")
ev(eb_mean, "Empirical-Bayes BetaBin x shrunk mean spend", "btyd")
ev(eb_clv, "Empirical-Bayes BetaBin x shrunk LOG spend", "btyd", "targets E[log1p] directly")

logcal(bg_clv, "BG/NBD x Gamma-Gamma + log-calibration", "btyd")
logcal(bg_clv_emp, "BG/NBD x empirical spend + log-calibration", "btyd")
logcal(eb_mean, "Empirical-Bayes BetaBin + log-calibration", "btyd")
logcal(eb_clv, "Empirical-Bayes BetaBin (log spend) + log-calibration", "btyd")

# save BTYD outputs as features for Tier 3
import pickle
feats = {}
for a in set(TRAIN_ANCHORS + [A, AH] + list(range(188, 260, 14))):
    try:
        par, x, t_x, T, nb, mv = bg(a)
        p, _ = eb_betabin(a)
        mlog, _ = eb_lognormal_spend(a)
        feats[a] = np.stack([bgnbd_expected(par, x, t_x, T, H), bgnbd_palive(par, x, t_x, T),
                             p, mlog], 1).astype(np.float32)
    except Exception as e:
        print("  skip anchor", a, e)
with open("model_out/btyd_feats.pkl", "wb") as f:
    pickle.dump(feats, f)
print(f"  BTYD features saved for {len(feats)} anchors")
print("Tier 2 done")
