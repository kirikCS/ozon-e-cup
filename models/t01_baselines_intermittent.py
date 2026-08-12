"""Tier 0 + Tier 1: baselines and intermittent-demand models.
All tunable parameters are fitted on TRAIN anchors and only then evaluated on the validation anchor.
"""
import sys
import numpy as np

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, TRAIN_ANCHORS, H, NU, N_DAYS, panel, target,
                    cohort_rule, rmsle, record, Timer, ds)

A, AH = VAL_ANCHOR, HON_ANCHOR
FIT_ANCHORS = TRAIN_ANCHORS[:4]          # 2025-12-15 .. 2025-10-30, closest to validation
G = panel("gmv")
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
wG = lambda a, b: csG[:, b + 1] - csG[:, a]

y_val = target(A)
y_hon = target(AH)
m_hon = cohort_rule(AH)

def ev(pred_fn, name, family, notes=""):
    with Timer() as t:
        pv = pred_fn(A)
        ph = pred_fn(AH)
    return record(name, family, rmsle(y_val, pv), rmsle(y_hon[m_hon], ph[m_hon]), t.dt, notes)

def fit_scalar(pred_of, grid):
    """pick the parameter minimising mean RMSLE across FIT_ANCHORS"""
    best, bp = 1e9, None
    for p in grid:
        s = np.mean([rmsle(target(a), pred_of(a, p)) for a in FIT_ANCHORS])
        if s < best: best, bp = s, p
    return bp

print(f"validation anchor {ds(A)} -> [{ds(A+1)}, {ds(A+H)}], {NU} users")
print(f"honest anchor     {ds(AH)} -> [{ds(AH+1)}, {ds(AH+H)}], cohort {m_hon.sum()}")
print(f"parameters fitted on anchors {[ds(a) for a in FIT_ANCHORS]}\n--- Tier 0: baselines ---")

ev(lambda a: np.zeros(NU), "const 0", "baseline")
c = fit_scalar(lambda a, p: np.full(NU, p), np.linspace(0, 40, 161))
ev(lambda a: np.full(NU, c), f"const optimal (c={c:.2f})", "baseline", "c fitted on train anchors")
ev(lambda a: wG(a - 29, a), "prev30 (naive AR = sample_submit)", "baseline")
ev(lambda a: wG(a - 59, a) / 2, "prev60 / 2", "baseline")
ev(lambda a: wG(a - 89, a) / 3, "prev90 / 3", "baseline")
ev(lambda a: wG(a - 179, a) / 6, "prev180 / 6", "baseline")
ev(lambda a: wG(a - 364, a) / 12.17, "prev365 / 12.17", "baseline")
ev(lambda a: wG(0, a) / ((a + 1) / 30), "lifetime rate per 30d", "baseline")
ev(lambda a: (wG(a - 29, a) + wG(a - 59, a) / 2 + wG(a - 89, a) / 3) / 3, "mean(prev30, prev60/2, prev90/3)", "baseline")

al = fit_scalar(lambda a, p: p * wG(a - 29, a), np.linspace(0, 1.5, 151))
ev(lambda a: al * wG(a - 29, a), f"alpha*prev30 (alpha={al:.2f})", "baseline", "alpha fitted on train")
al9 = fit_scalar(lambda a, p: p * wG(a - 89, a) / 3, np.linspace(0, 2, 201))
ev(lambda a: al9 * wG(a - 89, a) / 3, f"alpha*prev90/3 (alpha={al9:.2f})", "baseline", "alpha fitted on train")

# log-linear shrink, fitted on train anchors
best = (1e9, None)
for b in np.linspace(0.2, 1.2, 41):
    for cc in np.linspace(-0.5, 2.0, 51):
        s = np.mean([rmsle(target(a), np.expm1(b * np.log1p(wG(a - 29, a)) + cc)) for a in FIT_ANCHORS])
        if s < best[0]: best = (s, (b, cc))
bb, cc = best[1]
ev(lambda a: np.expm1(bb * np.log1p(wG(a - 29, a)) + cc),
   f"expm1({bb:.2f}*log1p(prev30)+{cc:.2f})", "baseline", "b,c fitted on train")

# same but on prev90/3
best2 = (1e9, None)
for b in np.linspace(0.2, 1.2, 41):
    for cc2 in np.linspace(-0.5, 2.0, 51):
        s = np.mean([rmsle(target(a), np.expm1(b * np.log1p(wG(a - 89, a) / 3) + cc2)) for a in FIT_ANCHORS])
        if s < best2[0]: best2 = (s, (b, cc2))
b2, c2 = best2[1]
ev(lambda a: np.expm1(b2 * np.log1p(wG(a - 89, a) / 3) + c2),
   f"expm1({b2:.2f}*log1p(prev90/3)+{c2:.2f})", "baseline", "b,c fitted on train")

# EWMA over 30-day blocks
def blocks(a, k=8):
    return np.stack([wG(a - 30 * i - 29, a - 30 * i) for i in range(k)], 1)
lam = fit_scalar(lambda a, p: (blocks(a) * (p ** np.arange(8))).sum(1) / (p ** np.arange(8)).sum(),
                 np.linspace(0.1, 0.99, 45))
w8 = lam ** np.arange(8); w8 = w8 / w8.sum()
ev(lambda a: blocks(a) @ w8, f"EWMA of 30d blocks (lambda={lam:.2f})", "baseline", "lambda fitted on train")
ev(lambda a: np.median(blocks(a, 6), axis=1), "median of last 6 blocks", "baseline")
ev(lambda a: np.exp(np.log1p(blocks(a, 6)).mean(1)) - 1, "geometric mean of last 6 blocks", "baseline")
ev(lambda a: wG(a + 1 - 365, a + 30 - 365), "seasonal naive (same window a year ago)", "baseline")
ev(lambda a: 0.5 * wG(a - 29, a) + 0.5 * wG(a + 1 - 365, a + 30 - 365), "0.5*prev30 + 0.5*seasonal naive", "baseline")

# ------------------------------------------------------------------ Tier 1
print("--- Tier 1: intermittent demand ---")

def croston(alpha, variant="classic", upto=N_DAYS - 1):
    """returns per-period demand rate at each anchor; vectorised over users"""
    z = np.zeros(NU); p = np.ones(NU); q = np.ones(NU); started = np.zeros(NU, bool)
    out = np.zeros((NU, upto + 1), np.float32)
    for t in range(upto + 1):
        y = G[:, t].astype(np.float64)
        nz = y > 0
        z[nz & ~started] = y[nz & ~started]; p[nz & ~started] = q[nz & ~started]
        upd = nz & started
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]
        started |= nz
        q[nz] = 1; q[~nz] += 1
        r = z / np.maximum(p, 1e-9)
        if variant == "sba": r = r * (1 - alpha / 2)
        out[:, t] = np.where(started, r, 0)
    return out

def tsb(alpha, beta, upto=N_DAYS - 1):
    z = np.zeros(NU); pr = np.zeros(NU); started = np.zeros(NU, bool)
    out = np.zeros((NU, upto + 1), np.float32)
    for t in range(upto + 1):
        y = G[:, t].astype(np.float64)
        nz = y > 0
        z[nz & ~started] = y[nz & ~started]
        upd = nz & started
        z[upd] = alpha * y[upd] + (1 - alpha) * z[upd]
        pr = np.where(nz, beta + (1 - beta) * pr, (1 - beta) * pr)
        started |= nz
        out[:, t] = np.where(started, pr * z, 0)
    return out

def adida(alpha, agg, upto=N_DAYS - 1):
    """aggregate into buckets of `agg` days, SES on bucket totals, rescale to 30 days"""
    out = np.zeros((NU, upto + 1), np.float32)
    lvl = np.zeros(NU); init = False
    for t in range(upto + 1):
        if (t + 1) % agg == 0:
            b = wG(t - agg + 1, t)
            lvl = b if not init else alpha * b + (1 - alpha) * lvl
            init = True
        out[:, t] = lvl / agg
    return out

CACHE_R = {}
def rate(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in CACHE_R:
        CACHE_R[key] = {"croston": lambda: croston(kw["alpha"], "classic"),
                        "sba": lambda: croston(kw["alpha"], "sba"),
                        "tsb": lambda: tsb(kw["alpha"], kw["beta"]),
                        "adida": lambda: adida(kw["alpha"], kw["agg"])}[kind]()
    return CACHE_R[key]

def fit_int(kind, grid):
    best, bp = 1e9, None
    for kw in grid:
        R = rate(kind, **kw)
        s = np.mean([rmsle(target(a), H * R[:, a]) for a in FIT_ANCHORS])
        if s < best: best, bp = s, kw
    return bp

g_a = [{"alpha": a} for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]]
p = fit_int("croston", g_a)
ev(lambda a: H * rate("croston", **p)[:, a], f"Croston classic (alpha={p['alpha']})", "intermittent", "alpha fitted on train")
p = fit_int("sba", g_a)
ev(lambda a: H * rate("sba", **p)[:, a], f"Croston SBA (alpha={p['alpha']})", "intermittent", "alpha fitted on train")
g_t = [{"alpha": a, "beta": b} for a in [0.05, 0.1, 0.2, 0.3] for b in [0.02, 0.05, 0.1, 0.2]]
p = fit_int("tsb", g_t)
ev(lambda a: H * rate("tsb", **p)[:, a], f"TSB (alpha={p['alpha']}, beta={p['beta']})", "intermittent", "fitted on train")
g_ad = [{"alpha": a, "agg": g} for a in [0.1, 0.2, 0.3, 0.5] for g in [30]]
p = fit_int("adida", g_ad)
ev(lambda a: H * rate("adida", **p)[:, a], f"ADIDA (alpha={p['alpha']}, agg=30)", "intermittent", "alpha fitted on train")
# IMAPA = mean of ADIDA over several aggregation levels
lv = [7, 14, 30, 60]
al_i = fit_scalar(lambda a, pa: H * np.mean([rate("adida", alpha=pa, agg=g)[:, a] for g in lv], 0), [0.1, 0.2, 0.3, 0.5])
ev(lambda a: H * np.mean([rate("adida", alpha=al_i, agg=g)[:, a] for g in lv], 0),
   f"IMAPA (levels {lv}, alpha={al_i})", "intermittent", "alpha fitted on train")

# shrunk variants: the raw rate optimises the conditional mean, RMSLE wants the log-mean
for kind, par, lab in [("tsb", fit_int("tsb", g_t), "TSB"), ("sba", fit_int("sba", g_a), "Croston SBA")]:
    R = rate(kind, **par)
    best = (1e9, None)
    for b in np.linspace(0.2, 1.2, 41):
        for cc3 in np.linspace(-0.5, 2.0, 51):
            s = np.mean([rmsle(target(a), np.expm1(b * np.log1p(H * R[:, a]) + cc3)) for a in FIT_ANCHORS])
            if s < best[0]: best = (s, (b, cc3))
    b3, c3 = best[1]
    ev(lambda a, R=R, b3=b3, c3=c3: np.expm1(b3 * np.log1p(H * R[:, a]) + c3),
       f"{lab} + log-calibration", "intermittent", f"b={b3:.2f}, c={c3:.2f} fitted on train")

print("\nTier 0-1 done")
