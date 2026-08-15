"""The control r6 was missing: is the warm specialist's +0.0014 anything but a second model?

r6 showed that adding a warm-trained arm to the pooled model buys +0.0011..+0.0019 beyond the
affine correction, consistently in both transfer directions. But `plain` — the pooled
formulation merely retrained on the warm subset — bought as much as `ratio30` and `mult`. When
every arm pays the same regardless of what it does, the payment is not for what it does.

The remaining explanation is the cheapest one: two models blend better than one, whatever the
second model is. The teammate already measured this once (7 seeds on identical features beat
the best single by +0.00526, and 7 RFE subsets beat it by +0.00486 — the same number), and it
closed the "different feature subsets per component" idea.

So the arm here is a SECOND POOLED MODEL with a different seed: same rows, same features, same
formulation, nothing special at all. It is blended by exactly the same procedure, fitted at one
anchor and applied frozen at the other, with the same affine control subtracted.

  seed_pool   second pooled model, different seed   <- the null
  warm_plain  pooled formulation on warm rows only  <- r6's arm, repeated for comparability
  warm_mult   gate x (clicks + personal click value)

If seed_pool pays the same as warm_plain, the warm split contributes nothing of its own and the
whole branch closes. If it pays visibly less, the difference is what the segment actually buys.
"""
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378)}
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True)
G = panel("gmv")
CART = panel("cart").astype(np.float32)

def vpc(a, k=6):
    bl = [(max(0, a - 30 * (i + 1) + 1), a - 30 * i) for i in range(k)]
    gm = np.stack([G[:, s:e + 1].sum(1) for s, e in bl], 1)
    cm = np.stack([CART[:, s:e + 1].sum(1) for s, e in bl], 1)
    ok = (cm > 0) & (gm > 0)
    lv = np.where(ok, np.log1p(gm) - np.log1p(cm), np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(lv, 1)
    n = ok.sum(1)
    pop = float(np.nanmedian(lv[ok])) if ok.any() else 0.0
    return np.where(n > 0, (np.nan_to_num(med) * n + pop * 2) / (n + 2), pop)

def warm_mask(a):
    bd = (G[:, :a + 1] > 0).sum(1)
    return bd > np.median(bd)

RES = {}
for proto, (TR, EV) in PROTO.items():
    print(f"[{proto}] {ds(EV)} ...", flush=True)
    t0 = time.time()
    X, _ = build_features(sorted(set(TR + [EV])), verbose=False)
    Xtr = np.concatenate([X[a] for a in TR], 0)
    ytr_raw = np.concatenate([target(a) for a in TR])
    ytr = np.log1p(ytr_raw)
    Wt = np.concatenate([warm_mask(a) for a in TR])
    rows = np.flatnonzero(Wt)

    base = lgb.LGBMRegressor(**dict(LGB, objective="regression",
                                    random_state=42)).fit(Xtr, ytr).predict(X[EV])
    arms = {}
    # НУЛЬ: второй пул, отличается только сидом — ничем больше
    arms["seed_pool"] = lgb.LGBMRegressor(**dict(LGB, objective="regression",
                                                 random_state=101)).fit(Xtr, ytr).predict(X[EV])
    arms["warm_plain"] = lgb.LGBMRegressor(**dict(LGB, objective="regression",
                                                  random_state=101)).fit(
        Xtr[rows], ytr[rows]).predict(X[EV])
    gate = lgb.LGBMClassifier(**dict(LGB, objective="binary", random_state=101)).fit(
        Xtr[rows], (ytr_raw[rows] > 0).astype(int))
    ctr = np.concatenate([np.log1p(CART[:, a + 1:a + 31].sum(1)) for a in TR])
    pos = rows[ytr_raw[rows] > 0]
    cm = lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=101)).fit(
        Xtr[pos], ctr[pos])
    arms["warm_mult"] = gate.predict_proba(X[EV])[:, 1] * np.clip(cm.predict(X[EV]) + vpc(EV), 0, 13)
    RES[proto] = (base, arms, warm_mask(EV), np.log1p(target(EV)), target(EV))
    print(f"    готово ({time.time()-t0:.0f}s)", flush=True)
    del Xtr, X

print("\n=== вклад руки сверх аффинной поправки, сегмент buyday > медианы ===")
print(f"{'фит -> прим.':18s} {'пул':>10} {'аффинный':>10} {'+рука':>10} "
      f"{'сверх аффинного':>16}  рука")
for src, dst in (("SELECT", "VAL"), ("VAL", "SELECT")):
    bs, as_, Ws, lys, _ = RES[src]
    bd, ad, Wd, lyd, _ = RES[dst]
    base = np.sqrt(np.mean((np.clip(bd, 0, 13)[Wd] - lyd[Wd]) ** 2))
    A1 = np.stack([bs[Ws], np.ones(int(Ws.sum()))], 1)
    w1 = np.linalg.lstsq(A1, lys[Ws], rcond=None)[0]
    B1 = np.stack([bd[Wd], np.ones(int(Wd.sum()))], 1)
    aff = np.sqrt(np.mean((np.clip(B1 @ w1, 0, 13) - lyd[Wd]) ** 2))
    for tag in ("seed_pool", "warm_plain", "warm_mult"):
        A2 = np.stack([bs[Ws], as_[tag][Ws], np.ones(int(Ws.sum()))], 1)
        w2 = np.linalg.lstsq(A2, lys[Ws], rcond=None)[0]
        B2 = np.stack([bd[Wd], ad[tag][Wd], np.ones(int(Wd.sum()))], 1)
        r = np.sqrt(np.mean((np.clip(B2 @ w2, 0, 13) - lyd[Wd]) ** 2))
        mark = "   <- НУЛЬ" if tag == "seed_pool" else ""
        print(f"{src+' -> '+dst:18s} {base:10.5f} {aff:10.5f} {r:10.5f} {aff-r:+16.5f}  "
              f"{tag} (вес {w2[1]:+.3f}){mark}")
print("\nчитать так: если warm_* не бьёт seed_pool, тёплый сплит не несёт ничего своего")
print("done")
