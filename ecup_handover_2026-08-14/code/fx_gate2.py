"""Gate power-tempering, honestly: gamma fitted on training-anchor OOF, applied frozen.

fx_gate.py showed the gate is overconfident in every decile at the eval anchor and that
the eval-anchor-optimal gamma=1.10 is worth -0.0022. That sweep was diagnostic cheating;
here gamma comes from leave-one-anchor-out OOF over the five TRAINING anchors only:
train clf+reg on 4 anchors, predict the held-out one, pool, fit gamma by MSLE grid.
The frozen gamma is then applied at the eval anchor.

Control for the slope confound: on the same OOF pool fit a scalar b for ly' = b*(pb*mu)
and apply it frozen the same way. If b matches gamma's gain, the effect is generic slope,
not a gate property (and the stack already handles slope exactly).

Caveat noted: adjacent training anchors' target windows overlap (28d spacing vs 30d
horizon), so the OOF is mildly optimistic; the decision rests on the frozen application
at the eval anchor and on VAL/HON sign agreement, not on the OOF numbers.

Usage:  python -u code/fx_gate2.py SELECT|VAL|HON
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, cohort_rule

PROTO = {
    "SELECT": ([318, 290, 262, 234, 206], 348, False),
    "VAL": ([348, 320, 292, 264, 236], 378, False),
    "HON": ([258, 230, 202, 174, 146], 288, True),
}

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)


def two_stage(X, tr, ev, seed=42):
    Xtr = np.concatenate([X[a] for a in tr], 0)
    lytr = np.concatenate([np.log1p(target(a)) for a in tr])
    pos = lytr > 0
    clf = lgb.LGBMClassifier(**LGB, random_state=seed)
    clf.fit(Xtr, pos.astype(np.int8))
    pb = clf.predict_proba(X[ev])[:, 1]
    reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=seed)
    reg.fit(Xtr[pos], lytr[pos])
    mu = np.clip(reg.predict(X[ev]), 0, None)
    return pb, mu


def main(proto):
    TR, EV, use_mask = PROTO[proto]
    anchors = sorted(set(TR + [EV]))
    X, _ = build_features(anchors)
    mask = cohort_rule(EV) if use_mask else None

    # OOF over training anchors
    oof_pb, oof_mu, oof_ly = [], [], []
    for t in TR:
        t0 = time.time()
        pb, mu = two_stage(X, [a for a in TR if a != t], t)
        oof_pb.append(pb); oof_mu.append(mu); oof_ly.append(np.log1p(target(t)))
        print(f"  OOF anchor {t} done ({time.time()-t0:.0f}s)", flush=True)
    pb = np.concatenate(oof_pb); mu = np.concatenate(oof_mu); ly = np.concatenate(oof_ly)

    def msle(pred_ly):
        return float(np.mean((ly - np.clip(pred_ly, 0, None)) ** 2))

    gammas = np.arange(0.90, 1.31, 0.025)
    gs = [msle(np.clip(pb, 1e-9, 1) ** g * mu) for g in gammas]
    g_star = float(gammas[int(np.argmin(gs))])
    bs = np.arange(0.90, 1.11, 0.01)
    bsc = [msle(b * pb * mu) for b in bs]
    b_star = float(bs[int(np.argmin(bsc))])
    print(f"OOF: gamma* {g_star:.3f}  slope b* {b_star:.3f}")

    # frozen application at the eval anchor
    pbe, mue = two_stage(X, TR, EV)
    y = target(EV)
    if mask is not None:
        y = y[mask]

    def sc(v):
        return rmsle(y, np.expm1(np.clip(v[mask] if mask is not None else v, 0, None)))

    base = sc(pbe * mue)
    gam = sc(np.clip(pbe, 1e-9, 1) ** g_star * mue)
    slp = sc(b_star * pbe * mue)
    print(f"{proto}: base {base:.5f}  gamma* {gam:.5f} ({gam-base:+.5f})  "
          f"slope* {slp:.5f} ({slp-base:+.5f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SELECT")
