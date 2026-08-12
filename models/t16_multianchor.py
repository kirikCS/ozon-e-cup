"""Multi-anchor selection: feature elimination and anchor weighting judged on several seasonal
phases at once, not on the January trough alone.

Why: four improvements in a row failed to transfer from the validation window to the leaderboard
(v2 features, RFE, seasonal anchors, dropping the calendar block). The common factor is that every
selection was made against the target window 2026-01-15..02-13 — the only seasonal trough of the
year — while the competition window sits on the 23 Feb / 8 Mar rise.

Harness: four evaluation anchors spanning trough, pre-NY peak and autumn. For each, a model is
trained on the four anchors preceding it (rolling origin, no leakage), and the selection criterion
is the mean RMSLE across all four.

  eval 378 -> target 2026-01-15..02-13   trough,   target/prev30 = 0.83
  eval 318 -> target 2025-11-16..12-15   peak,     1.11
  eval 288 -> target 2025-10-17..11-15             1.01
  eval 252 -> target 2025-09-11..10-10             1.09

Note on the weighting: in this data recency and seasonal proximity coincide — the only
seasonally-matched historical anchors sit near day 43 with 44 days of history. So the decay below
is honestly a decay in anchor age, not a separable seasonal effect.
"""
import json
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "models")
import btyd_lib as BL
from common import H, N_DAYS, NU, panel, target, rmsle, build_features, OUT, CACHE, ds
from features_v2 import build_v2

PRED = N_DAYS - 1
EVAL = [378, 318, 288, 252]
TRAIN_OF = {e: [e - 30, e - 58, e - 86, e - 114] for e in EVAL}
TR_FINAL = [378, 350, 322, 294]
ALL_A = sorted(set(EVAL + [a for v in TRAIN_OF.values() for a in v] + TR_FINAL + [PRED]))
TAUS = [None, 200.0, 120.0, 80.0, 50.0]          # None = uniform weights
KEEP_FRACS = [1.0, 0.75, 0.55, 0.40, 0.28, 0.20, 0.14]
EL = 2.3284
print(f"anchors {ALL_A}")

# ------------------------------------------------------------------ BTYD top-up
G, V = panel("gmv"), panel("visit")
BUY = G > 0
csG = np.concatenate([np.zeros((NU, 1)), G.astype(np.float64)], 1).cumsum(1)
csB = np.concatenate([np.zeros((NU, 1), np.int64), BUY.astype(np.int64)], 1).cumsum(1)
wG = lambda a, b: csG[:, max(b, 0) + 1] - csG[:, max(a, 0)]
SUBS = np.random.default_rng(0).choice(NU, 25000, replace=False)
K_, KAP, DEC = 10, 0.5, 0.7

def rfm(a):
    sub = V[:, :a + 1] > 0
    first = np.argmax(sub, 1)
    bs = BUY[:, :a + 1]
    has = bs.any(1)
    last = a - np.argmax(bs[:, ::-1], 1)
    nb = csB[:, a + 1]
    x = np.maximum(nb - 1, 0).astype(np.float64)
    T = (a - first).astype(np.float64)
    return x, np.minimum(np.where(has, last - first, 0).astype(np.float64), T), T, nb, \
           np.where(nb > 0, csG[:, a + 1] / np.maximum(nb, 1), 0.0)

def eb_lb(a):
    L = np.stack([np.log1p(wG(a - 30 * i - 29, a - 30 * i)) for i in range(K_)], 1)
    w = DEC ** np.arange(K_)
    pop = (L * w).sum() / (w.sum() * NU)
    return np.expm1(((L * w).sum(1) + KAP * pop) / (w.sum() + KAP))

BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
missing = [a for a in ALL_A if a not in BT]
if missing:
    print(f"BTYD top-up for {missing}")
    for a in sorted(missing):
        x, tx, T, nb, mv = rfm(a)
        par = BL.bgnbd_fit(x[SUBS], tx[SUBS], T[SUBS]); gp = BL.gg_fit(nb, mv)
        BT[a] = np.stack([BL.bgnbd_expected(par, x, tx, T, H), BL.bgnbd_palive(par, x, tx, T),
                          BL.gg_expected(gp, nb, mv), (csB[:, a + 1] + 1.0) / (a + 1 + 20.0),
                          np.log1p(eb_lb(a))], 1).astype(np.float32)
    pickle.dump(BT, open(OUT / "btyd_feats.pkl", "wb"))
    print("  done")

X1, N1 = build_features(ALL_A)
BT_NAMES = ["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"]
for a in ALL_A:
    X1[a] = np.concatenate([X1[a], BT[a]], 1)
X2, N2 = build_v2(ALL_A)
NAMES = np.array(N1 + BT_NAMES + N2)
X = {a: np.concatenate([X1[a], X2[a]], 1) for a in ALL_A}
del X1, X2
F = X[PRED].shape[1]
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"features {F}")
for e in EVAL:
    print(f"  eval {ds(e)} -> {ds(e+1)}..{ds(e+H)}, train {[ds(a) for a in TRAIN_OF[e]]}")

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def weights(anchors, ref, tau):
    if tau is None:
        return np.ones(len(anchors))
    return np.exp(-np.array([(ref - a) for a in anchors], dtype=float) / tau)

def fit_lgb(Xtr, lytr, sw, Xev):
    m = lgb.LGBMRegressor(**dict(LGB, objective="regression"))
    m.fit(Xtr, lytr, sample_weight=sw)
    return np.clip(np.expm1(m.predict(Xev)), 0, None), m.booster_.feature_importance("gain")

def fit_cat(Xtr, lytr, sw, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr, sample_weight=sw)
    return np.clip(np.expm1(m.predict(Xev)), 0, None), np.asarray(m.get_feature_importance())

def evaluate(fn, idx, tau, evals=EVAL):
    """mean RMSLE over the evaluation anchors; also returns the averaged gain profile"""
    scores, gains = [], np.zeros(len(idx))
    for e in evals:
        tr = TRAIN_OF[e]
        Xtr = np.concatenate([X[a][:, idx] for a in tr], 0)
        ly = np.concatenate([np.log1p(Y[a]) for a in tr])
        wv = weights(tr, e, tau)
        sw = np.concatenate([np.full(NU, wv[i]) for i in range(len(tr))])
        p, g = fn(Xtr, ly, sw, X[e][:, idx])
        del Xtr
        scores.append(rmsle(Y[e], p))
        gains += g / max(g.sum(), 1e-9)
    return float(np.mean(scores)), scores, gains

results = {}

# ------------------------------------------------------- phase 1: anchor weighting
print("\n=== фаза 1: затухание веса обучающих якорей ===")
allc = np.arange(F)
for mname, fn in (("LightGBM L2", fit_lgb), ("CatBoost", fit_cat)):
    print(f"\n{mname}")
    row = []
    for tau in TAUS:
        t0 = time.time()
        mean, sc, _ = evaluate(fn, allc, tau)
        row.append({"tau": tau, "mean": mean, "per_anchor": sc})
        lab = "равномерно" if tau is None else f"tau={tau:.0f}д"
        print(f"  {lab:12s} среднее {mean:.5f}   " +
              " ".join(f"{ds(e)[5:]}:{s:.4f}" for e, s in zip(EVAL, sc)) + f"   ({time.time()-t0:.0f}s)")
    best = min(row, key=lambda r: r["mean"])
    results[f"weights::{mname}"] = row
    lab = "равномерно" if best["tau"] is None else f"tau={best['tau']:.0f}д"
    print(f"  лучший: {lab}  среднее {best['mean']:.5f}")

BEST_TAU = {m: min(results[f"weights::{m}"], key=lambda r: r["mean"])["tau"]
            for m in ("LightGBM L2", "CatBoost")}
print(f"\nвыбранные tau: {BEST_TAU}")

# ------------------------------------------------------- phase 2: multi-anchor RFE
print("\n=== фаза 2: RFE по среднему из четырёх якорей ===")
selected = {}
for mname, fn in (("CatBoost", fit_cat), ("LightGBM L2", fit_lgb)):
    tau = BEST_TAU[mname]
    print(f"\n{mname} (tau={'равномерно' if tau is None else int(tau)})")
    idx = np.arange(F); imp = None; hist = []
    for frac in KEEP_FRACS:
        k = max(8, int(round(F * frac)))
        if k < len(idx):
            idx = idx[np.argsort(imp)[::-1][:k]]
        t0 = time.time()
        mean, sc, imp = evaluate(fn, idx, tau)
        hist.append({"n": int(len(idx)), "mean": mean, "per_anchor": sc,
                     "idx": idx.tolist() if len(idx) <= 120 else None})
        print(f"  {len(idx):4d} признаков  среднее {mean:.5f}   " +
              " ".join(f"{s:.4f}" for s in sc) + f"   ({time.time()-t0:.0f}s)")
    best = min(hist, key=lambda r: r["mean"])
    results[f"rfe::{mname}"] = hist
    selected[mname] = {"n": best["n"], "idx": best["idx"], "tau": tau, "mean": best["mean"]}
    print(f"  лучший: {best['n']} признаков, среднее {best['mean']:.5f} "
          f"(полный набор: {hist[0]['mean']:.5f}, дельта {hist[0]['mean']-best['mean']:+.5f})")

json.dump({"results": results,
           "selected": {k: {"n": v["n"], "tau": v["tau"], "mean": v["mean"],
                            "names": NAMES[v["idx"]].tolist() if v["idx"] else None}
                        for k, v in selected.items()}},
          open(OUT / "multianchor.json", "w"), ensure_ascii=False, indent=1)

# ------------------------------------------------------- phase 3: build submissions
print("\n=== фаза 3: сборка компонентов на якорь 2026-02-13 ===")
uids = np.load(CACHE / "uids.npy")
for mname, fn, fname in (("CatBoost", fit_cat, "ma_catboost"), ("LightGBM L2", fit_lgb, "ma_lgbm")):
    s = selected[mname]
    idx = np.array(s["idx"]) if s["idx"] else np.arange(F)
    tau = s["tau"]
    Xtr = np.concatenate([X[a][:, idx] for a in TR_FINAL], 0)
    ly = np.concatenate([np.log1p(Y[a]) for a in TR_FINAL])
    wv = weights(TR_FINAL, PRED, tau)
    sw = np.concatenate([np.full(NU, wv[i]) for i in range(len(TR_FINAL))])
    p, _ = fn(Xtr, ly, sw, X[PRED][:, idx])
    del Xtr
    lp = np.log1p(np.clip(np.nan_to_num(p), 0, None)); lp = lp - lp.mean() + EL
    q = np.clip(np.expm1(lp), 0, None)
    with open(f"submissions/{fname}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, q):
            f.write(f"{u},{v:.6f}\n")
    print(f"  {fname}.csv: {len(idx)} признаков, tau={tau}, mean={q.mean():.2f}, "
          f"p50={np.percentile(q,50):.2f}, p99={np.percentile(q,99):.0f}")
print("\ndone")
