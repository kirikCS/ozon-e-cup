"""r1 and r5 restricted to warm clients — with the two defects of the earlier runs repaired.

Why restrict. Both ideas failed for reasons that are specific to cold clients:

  r1  with init_score = log1p(prev30) the residual target is the FULL log1p(Y) for the 43.7%
      whose prev30 is zero, and a small mean-reverting deviation for everyone else. One
      function had to cover two regimes. Dropping the cold rows removes the mixture.
  r5  clicks do not vanish when purchases do, so the multiplicative model predicted positive
      GMV for the 46% who buy nothing. On warm clients that failure mode is much weaker.

Two repairs to how the idea is tested, both mattering more than the restriction itself:

  BLEND, DO NOT ROUTE. r2 hard-switched the prediction on a segment, which makes the specialist
  carry its whole half alone and cost +0.0008. Here the warm model is an extra component and
  the weights are solved — a blend cannot lose by construction, so the routing penalty never
  appears. The blend weights are fitted at one anchor and applied FROZEN at another, the
  transfer discipline that killed fx_shrink.

  MULTIPLY THE CONDITIONAL PART ONLY. E[log1p y] = P(y>0) * E[log1p y | y>0] is exact, so
  clicks x value-per-click estimates the SECOND factor, with the gate coming from a classifier.
  That is the direct repair of the diagnosed defect rather than a restriction around it.

Arms, all trained on warm rows only, all evaluated on warm clients:
  plain     same formulation as the pooled model — isolates "fewer rows" from "new formulation"
  ratio30   init_score = log1p(prev30)
  ratioeb   init_score = EB-shrunk log-block mean
  mult      gate classifier x (predicted log clicks + shrunk personal value-per-click)
"""
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

PROTO = {"SELECT": ([318, 290, 262, 234, 206], 348),
         "VAL": ([348, 320, 292, 264, 236], 378)}
SEEDS = (42, 101)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True)
G = panel("gmv")
CART = panel("cart").astype(np.float32)

def prev30(a):
    return G[:, max(0, a - 29):a + 1].sum(1)

def eb_off(a):
    B = np.stack([np.log1p(G[:, max(0, a - 30 * k + 1):a - 30 * (k - 1) + 1].sum(1))
                  for k in range(1, 7)], 1)
    mu, v = B.mean(1), B.var(1)
    gm, gv = mu.mean(), mu.var()
    sh = gv / np.maximum(gv + v / 6.0, 1e-9)
    return sh * mu + (1 - sh) * gm

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

def warm_masks(a):
    bd = (G[:, :a + 1] > 0).sum(1)
    return {"prev30>0": prev30(a) > 0, "buyday>медианы": bd > np.median(bd)}

RES = {}
for proto, (TR, EV) in PROTO.items():
    print(f"\n=== {proto}: обучение {TR} -> оценка {ds(EV)} ===", flush=True)
    X, _ = build_features(sorted(set(TR + [EV])), verbose=False)
    Xtr = np.concatenate([X[a] for a in TR], 0)
    ytr_raw = np.concatenate([target(a) for a in TR])
    ytr = np.log1p(ytr_raw)
    Xev, yev = X[EV], target(EV)
    lyev = np.log1p(yev)

    pooled = np.mean([lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=s))
                      .fit(Xtr, ytr).predict(Xev) for s in SEEDS], 0)
    wm_tr = {k: np.concatenate([warm_masks(a)[k] for a in TR]) for k in warm_masks(EV)}
    wm_ev = warm_masks(EV)

    for wname in wm_ev:
        Wt, We = wm_tr[wname], wm_ev[wname]
        rows = np.flatnonzero(Wt)
        base_w = np.sqrt(np.mean((np.clip(pooled, 0, 13)[We] - lyev[We]) ** 2))
        print(f"\n  сегмент «{wname}»: доля {We.mean():.3f}, "
              f"тёплых строк в обучении {len(rows)}")
        print(f"  {'рука':10s} {'RMSLE на тёплых':>16} {'к пулу':>9} {'corr с пулом':>13}")
        print(f"  {'пул':10s} {base_w:16.5f} {0.0:+9.5f} {1.0:13.5f}")
        arms = {}
        for tag in ("plain", "ratio30", "ratioeb", "mult"):
            ps = []
            for s in SEEDS:
                if tag == "mult":
                    gate = lgb.LGBMClassifier(**dict(LGB, objective="binary", random_state=s))
                    gate.fit(Xtr[rows], (ytr_raw[rows] > 0).astype(int))
                    pg = gate.predict_proba(Xev)[:, 1]
                    ctr = np.concatenate([np.log1p(CART[:, a + 1:a + 31].sum(1)) for a in TR])
                    pos = rows[ytr_raw[rows] > 0]
                    cm = lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=s))
                    cm.fit(Xtr[pos], ctr[pos])
                    ps.append(pg * np.clip(cm.predict(Xev) + vpc(EV), 0, 13))
                    continue
                if tag == "plain":
                    otr = np.zeros(len(ytr)); oev = np.zeros(NU)
                elif tag == "ratio30":
                    otr = np.concatenate([np.log1p(prev30(a)) for a in TR]); oev = np.log1p(prev30(EV))
                else:
                    otr = np.concatenate([eb_off(a) for a in TR]); oev = eb_off(EV)
                m = lgb.LGBMRegressor(**dict(LGB, objective="regression", random_state=s))
                m.fit(Xtr[rows], (ytr - otr)[rows])
                ps.append(m.predict(Xev) + oev)
            p = np.mean(ps, 0)
            arms[tag] = p
            rw = np.sqrt(np.mean((np.clip(p, 0, 13)[We] - lyev[We]) ** 2))
            print(f"  {tag:10s} {rw:16.5f} {rw-base_w:+9.5f} "
                  f"{np.corrcoef(p[We], pooled[We])[0,1]:13.5f}")
        RES[(proto, wname)] = (pooled, arms, We, lyev)
    del Xtr, X

print("\n\n=== бленд вместо маршрутизации: веса с одного якоря, применение на другом ===")
for wname in ("prev30>0", "buyday>медианы"):
    print(f"\nсегмент «{wname}»")
    print(f"  {'фит -> прим.':18s} {'пул':>10} {'+рука':>10} {'выигрыш':>10}  рука")
    for src, dst in (("SELECT", "VAL"), ("VAL", "SELECT")):
        ps_, as_, Ws, lys = RES[(src, wname)]
        pd_, ad_, Wd, lyd = RES[(dst, wname)]
        base = np.sqrt(np.mean((np.clip(pd_, 0, 13)[Wd] - lyd[Wd]) ** 2))
        # КОНТРОЛЬ: аффинная поправка одного только пула, подобранная так же и на том же якоре.
        # Всё, что рука даёт сверх этого числа, и есть её вклад; всё до него — калибровка
        # уровня и наклона, которая в продакшне делается отдельно и точнее.
        Aa = np.stack([ps_[Ws], np.ones(int(Ws.sum()))], 1)
        wa = np.linalg.lstsq(Aa, lys[Ws], rcond=None)[0]
        Ba = np.stack([pd_[Wd], np.ones(int(Wd.sum()))], 1)
        aff = np.sqrt(np.mean((np.clip(Ba @ wa, 0, 13) - lyd[Wd]) ** 2))
        print(f"  {src+' -> '+dst:18s} {base:10.5f} {aff:10.5f} {base-aff:+10.5f}  "
              f"АФФИННЫЙ КОНТРОЛЬ (только пул)")
        for tag in ("plain", "ratio30", "ratioeb", "mult"):
            A = np.stack([ps_[Ws], as_[tag][Ws], np.ones(int(Ws.sum()))], 1)
            w = np.linalg.lstsq(A, lys[Ws], rcond=None)[0]
            B = np.stack([pd_[Wd], ad_[tag][Wd], np.ones(int(Wd.sum()))], 1)
            r = np.sqrt(np.mean((np.clip(B @ w, 0, 13) - lyd[Wd]) ** 2))
            print(f"  {src+' -> '+dst:18s} {base:10.5f} {r:10.5f} {base-r:+10.5f}  {tag}"
                  f"   (вес руки {w[1]:+.3f}, сверх аффинного {aff-r:+.5f})")
print("\ndone")
