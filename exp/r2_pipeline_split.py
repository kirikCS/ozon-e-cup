"""Split the PIPELINE, not the models — and split it asymmetrically.

fx_segment_models tested the symmetric version: two specialists, each trained on its own
segment. It lost (+0.0035) for a clear reason — both branches gave up the cross-segment rows,
and the cold branch is precisely the one that cannot afford to.

The asymmetric version has not been tested and does not have that defect:

    high-activity users  ->  a model trained ONLY on high-activity rows
    cool users           ->  a model trained on ALL rows

The cold branch keeps every row it had, so it loses nothing. The active branch specialises,
and it is the branch that can afford to: it has the most events per client and the least need
to borrow structure from anyone else. If specialisation ever pays here, it pays this way.

Controls, because any routing scheme adds parameters and will look good unaided:

  base       one pooled model everywhere — the reference
  asym       the arm above
  reverse    cold specialises, active pooled — if this also "wins", routing per se is the
             effect and the direction is meaningless
  random     asym on a random segment of the same size — the null

Four activity thresholds, three seeds each.
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

def buydays(a):
    return (G[:, :a + 1] > 0).sum(1)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ly_tr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
y_ev = target(SEL_EV)
Xev = X[SEL_EV]
BD = {a: buydays(a) for a in anchors}

def fit_pred(rows, seed):
    m = lgb.LGBMRegressor(**dict(LGB, random_state=seed))
    m.fit(Xtr[rows], ly_tr[rows]) if rows is not None else m.fit(Xtr, ly_tr)
    return m.predict(Xev)

print(f"=== асимметричное разделение пайплайна, оценка {ds(SEL_EV)} ===")
pool = {sd: fit_pred(None, sd) for sd in SEEDS}
base = np.mean([rmsle(y_ev, np.expm1(pool[sd])) for sd in SEEDS])
print(f"общий пул (референс): {base:.5f}   "
      f"сиды " + ", ".join(f"{rmsle(y_ev, np.expm1(pool[sd])):.5f}" for sd in SEEDS) + "\n")

def run(tag, hi_ev, hi_tr, specialise_hi):
    """specialise_hi=True: активные едут к специалисту, прохладные к общему пулу"""
    sel = hi_tr if specialise_hi else ~hi_tr
    if sel.sum() < 50_000:
        return None
    scores = []
    for sd in SEEDS:
        spec = fit_pred(np.flatnonzero(sel), sd)
        route = hi_ev if specialise_hi else ~hi_ev
        p = np.where(route, spec, pool[sd])
        scores.append(rmsle(y_ev, np.expm1(np.clip(p, 0, 13))))
    return float(np.mean(scores)), scores

rng = np.random.default_rng(0)
print(f"{'порог':>22} {'доля актив.':>11} {'асимм.':>9} {'к базе':>9} "
      f"{'реверс':>9} {'нуль(случ.)':>12}")
for qq in (50, 66, 80, 90):
    thr = np.percentile(BD[SEL_EV], qq)
    hi_ev = BD[SEL_EV] > thr
    hi_tr = np.concatenate([BD[a] > np.percentile(BD[a], qq) for a in SEL_TR])
    a_ = run("asym", hi_ev, hi_tr, True)
    r_ = run("rev", hi_ev, hi_tr, False)
    share = float(hi_ev.mean())
    nulls = []
    for _ in range(3):
        pe = rng.permutation(NU)
        rnd_ev = hi_ev[pe]
        rnd_tr = np.concatenate([hi_tr[i * NU:(i + 1) * NU][rng.permutation(NU)]
                                 for i in range(len(SEL_TR))])
        n_ = run("rnd", rnd_ev, rnd_tr, True)
        if n_:
            nulls.append(n_[0])
    print(f"{'buyday_all > p'+str(qq):>22} {share:11.3f} {a_[0]:9.5f} {a_[0]-base:+9.5f} "
          f"{r_[0]:9.5f} {np.mean(nulls):12.5f}")
    print(f"{'':22s} {'':11s} сиды асимм.: " + ", ".join(f"{v:.5f}" for v in a_[1]))
print("\nчитать так: асимметричная схема полезна, только если бьёт И базу, И реверс, И нуль")
print("done")
