"""Predict the coefficient to the previous month instead of the level.

In log space a ratio is a difference, so "predict Y/prev30" is exactly "predict
log1p(Y) - log1p(prev30) and add the offset back". For a tree that is not the no-op it looks
like: LightGBM's init_score makes the offset exact, which changes what the splits have to
explain — the model no longer spends capacity reproducing the level, only the deviation from
it. For a linear model it is a genuine reparametrisation.

The EDA motivates it and warns about it at the same time: target/prev30 falls monotonically
from 5.84 in the bottom decile to 0.46 in the top, so the ratio is strongly mean-reverting —
good, that is structure a model can use — but 46% of clients have prev30 = 0, where the offset
is log1p(0) = 0 and the reparametrisation does nothing at all. So the effect, if any, lives
entirely in the active half.

Five offsets, from the crudest to the one that already knows about mean reversion:

  none        the current formulation, reference
  prev30      the literal "coefficient to the previous month"
  prev90/3    same but averaged over three months, less noisy
  eb          empirical-Bayes shrunk mean of log1p over six 30-day blocks — the zero-parameter
              prior that already scores 1.84 on its own
  table       not a tree at all: the per-decile median ratio, i.e. the mean-reversion curve
              used directly as a model. The purely statistical version of the idea.

Every tree arm runs 3 seeds; the screen's noise band is about +-0.0004.
"""
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = (42, 101, 777)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")
G = panel("gmv")

def w(a, n):
    return G[:, max(0, a - n + 1):a + 1].sum(1)

def off_none(a):
    return np.zeros(NU, np.float64)

def off_prev30(a):
    return np.log1p(w(a, 30))

def off_prev90(a):
    return np.log1p(w(a, 90) / 3.0)

def off_eb(a):
    B = np.stack([np.log1p(G[:, max(0, a - 30 * k + 1):a - 30 * (k - 1) + 1].sum(1))
                  for k in range(1, 7)], 1)
    mu, v = B.mean(1), B.var(1)
    gm, gv = mu.mean(), mu.var()
    sh = gv / np.maximum(gv + v / 6.0, 1e-9)
    return sh * mu + (1 - sh) * gm

OFFS = {"none": off_none, "prev30": off_prev30, "prev90/3": off_prev90, "eb": off_eb}

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ly_tr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
y_ev = target(SEL_EV)
ly_ev = np.log1p(y_ev)
Xev = X[SEL_EV]

zero_share = float((w(SEL_EV, 30) == 0).mean())
print(f"=== цель как коэффициент к предыдущему месяцу, оценка {ds(SEL_EV)} ===")
print(f"доля клиентов с prev30 = 0 (где смещение тождественно ноль): {zero_share:.3f}\n")

print(f"{'смещение':12s} {'RMSLE':>9} {'к none':>9}   {'на активных':>12} {'на нулевых':>11}")
act = w(SEL_EV, 30) > 0
res = {}
base = None
for tag, f in OFFS.items():
    otr = np.concatenate([f(a) for a in SEL_TR])
    oev = f(SEL_EV)
    ps = []
    for sd in SEEDS:
        m = lgb.LGBMRegressor(**dict(LGB, random_state=sd))
        m.fit(Xtr, ly_tr - otr)
        ps.append(m.predict(Xev) + oev)
    p = np.mean(ps, 0)
    res[tag] = p
    r = rmsle(y_ev, np.expm1(np.clip(p, 0, 13)))
    if base is None:
        base = r
    ra = np.sqrt(np.mean((np.clip(p, 0, 13)[act] - ly_ev[act]) ** 2))
    rz = np.sqrt(np.mean((np.clip(p, 0, 13)[~act] - ly_ev[~act]) ** 2))
    sds = [rmsle(y_ev, np.expm1(np.clip(q + oev, 0, 13))) for q in
           [ps[i] - oev for i in range(len(ps))]]
    print(f"{tag:12s} {r:9.5f} {r-base:+9.5f}   {ra:12.5f} {rz:11.5f}"
          f"   сиды " + ", ".join(f"{v:.5f}" for v in sds))

# ---- чисто статистическая версия: кривая регрессии к среднему как модель
print("\n--- таблица медианных коэффициентов (без обучения) ---")
q_tr = np.concatenate([off_prev30(a) for a in SEL_TR])
r_tr = ly_tr - q_tr
edges = np.percentile(q_tr[q_tr > 0], np.arange(5, 100, 5))
def binof(v):
    return np.where(v > 0, np.searchsorted(edges, v) + 1, 0)
btr = binof(q_tr)
med = np.array([np.median(r_tr[btr == b]) if (btr == b).sum() > 50 else 0.0
                for b in range(len(edges) + 2)])
p_tab = off_prev30(SEL_EV) + med[binof(off_prev30(SEL_EV))]
print(f"  RMSLE {rmsle(y_ev, np.expm1(np.clip(p_tab,0,13))):.5f}   "
      f"(для сравнения: EB-приор без обучения даёт ~1.845)")
print(f"  медианный коэффициент по децилям prev30: "
      + ", ".join(f"{np.expm1(med[b]):.2f}" for b in range(1, len(edges) + 2, 4)))

print("\nвзаимные корреляции вариантов:")
ks = list(res)
for a in ks:
    print(f"  {a:10s} " + " ".join(f"{np.corrcoef(res[a],res[b])[0,1]:.5f}" for b in ks))
print("\ndone")
