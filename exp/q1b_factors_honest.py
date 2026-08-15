"""q1 gave -0.0198, which is fifty times any real find in this project. That is a leak alarm,
not a result. The cheapest explanation, and it is fatal:

    R = A - A.mean(axis=1, keepdims=True) - ...

centres each user by their mean over ALL 409 days — the target window included. A user who
buys a lot in days a+1..a+30 gets a larger row mean, so their centred history is pushed down
by an amount that encodes the very thing being predicted. Buy-day counts correlate 0.63 with
the target, so the leak is strong.

This rebuild uses only days <= anchor, everywhere:

  window     the last 182 days BEFORE the anchor, in relative coordinates
  centring   row and column means of that window only
  factors    pooled day-day covariance over the TRAINING anchors' windows only
  loadings   projection of the anchor's own history window on those factors

Three arms, because the honest version still has two competing explanations:

  factors    the real eigenvectors — "clients share latent external drivers"
  random     the same number of random projections of the same window — controls for
             "any linear read of the raw daily shape helps, factor structure irrelevant"
  shuffled   real loadings permuted across users — destroys the client-factor link, keeps
             the marginal distribution; must give zero
"""
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
W = 182
K = 12

A = (panel("gmv") > 0).astype(np.float32)

def hist_window(a):
    """double-centred activity over the 182 days ending AT the anchor — history only"""
    lo = max(0, a - W + 1)
    Z = A[:, lo:a + 1].copy()
    if Z.shape[1] < W:
        Z = np.concatenate([np.zeros((NU, W - Z.shape[1]), np.float32), Z], 1)
    Z -= Z.mean(1, keepdims=True)
    Z -= Z.mean(0, keepdims=True)
    sd = Z.std(1, keepdims=True)
    return Z / np.maximum(sd, 1e-6)

print("=== честные факторы: окно строго до якоря ===")
t0 = time.time()
Cd = np.zeros((W, W), np.float64)
for a in SEL_TR:                      # оценочный якорь в оценку факторов не входит
    Zt = hist_window(a)
    Cd += (Zt.T @ Zt) / NU
Cd /= len(SEL_TR)
ev, V = np.linalg.eigh(Cd)
ev, V = ev[::-1], V[:, ::-1]
r = W / NU
hi = (1 + np.sqrt(r)) ** 2
print(f"полоса Марченко-Пастура [{(1-np.sqrt(r))**2:.4f}, {hi:.4f}]")
print("топ-15: " + " ".join(f"{v:.3f}" for v in ev[:15]))
print(f"выше полосы: {int((ev > hi).sum())} из {W}   ({time.time()-t0:.0f}s)")

F = V[:, :K]
rng = np.random.default_rng(0)
Frnd = rng.normal(size=(W, K)) / np.sqrt(W)

def loads(a, basis):
    return (hist_window(a) @ basis).astype(np.float32)

import lightgbm as lgb
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")
anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
yev = target(SEL_EV)

def arm(tag, maker):
    Xtr = np.concatenate([np.concatenate([X[a], maker(a)], 1) if maker else X[a]
                          for a in SEL_TR], 0)
    Xev = np.concatenate([X[SEL_EV], maker(SEL_EV)], 1) if maker else X[SEL_EV]
    s = [rmsle(yev, np.expm1(lgb.LGBMRegressor(**dict(LGB, random_state=sd))
                             .fit(Xtr, ytr).predict(Xev))) for sd in (42, 101, 777)]
    del Xtr
    print(f"  {tag:34s} {np.mean(s):.5f}   сиды " + ", ".join(f"{v:.5f}" for v in s))
    return float(np.mean(s))

print(f"\n=== {K} колонок сверх фичестора, 3 сида ===")
b = arm("база (165 признаков)", None)
f = arm("реальные факторы", lambda a: loads(a, F))
rd = arm("случайные проекции того же окна", lambda a: loads(a, Frnd))
perm = rng.permutation(NU)
sh = arm("факторы, перемешанные по клиентам", lambda a: loads(a, F)[perm])
print(f"\nэффект факторов сверх базы:        {f-b:+.5f}")
print(f"эффект случайных проекций:         {rd-b:+.5f}")
print(f"эффект перемешанных (должен быть 0): {sh-b:+.5f}")
print(f"\nсобственный вклад факторной структуры сверх произвольной проекции: {f-rd:+.5f}")
print("\ndone")
