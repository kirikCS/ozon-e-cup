"""Count-mixture decomposition: E[log1p S | x] = sum_k P(K=k|x) * E[log1p S | K=k, x].

K = number of buy-days in the target window (0..9+). The two-stage uses only the K>0 cut;
days-with-purchase is the strongest single signal in the data (r = 0.630) and has never
been the conditioning variable. Given K, log1p(sum) is tightly concentrated (log k + log
basket), so the conditional regressor is low-variance; the multiclass head brings a
different loss geometry — the stack's only remaining currency.

First cut: single seed on SELECT, standalone vs the two-stage, then the absorption
control (blend with the 9-seed two-stage ensemble on half/half splits) — HL-Gauss died
exactly there, so that contrast decides.

Usage:  python -u code/fx_kmix.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import panel, target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
KMAX = 9          # classes 0..8 and 9+ pooled

LGB = dict(num_leaves=63, min_child_samples=100, max_bin=63, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=23,
           verbose=-1, force_col_wise=True)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors)
G = panel("gmv")
BUY = G > 0


def kcount(a):
    return np.minimum(BUY[:, a + 1:a + 31].sum(axis=1), KMAX)


lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
ktr = np.concatenate([kcount(a) for a in SEL_TR]).astype(np.int32)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
Xev = X[SEL_EV]
yev = target(SEL_EV)

t0 = time.time()
clf = lgb.LGBMClassifier(objective="multiclass", num_class=KMAX + 1, n_estimators=250,
                         learning_rate=0.12, **LGB, random_state=42)
clf.fit(Xtr, ktr)
PK = clf.predict_proba(Xev)
print(f"P(K) head fitted ({time.time()-t0:.0f}s)", flush=True)

# conditional value regressor: train on positive rows with K appended
t0 = time.time()
pos = ktr > 0
Xtr_pos = np.concatenate([Xtr[pos], ktr[pos, None].astype(np.float32)], 1)
reg = lgb.LGBMRegressor(objective="regression", n_estimators=600, learning_rate=0.08,
                        **LGB, random_state=42)
reg.fit(Xtr_pos, lytr[pos])
print(f"value head fitted ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
lp = np.zeros(len(yev))
for k in range(1, KMAX + 1):
    Xk = np.concatenate([Xev, np.full((len(yev), 1), k, np.float32)], 1)
    lp += PK[:, k] * np.clip(reg.predict(Xk), 0, None)
print(f"decode done ({time.time()-t0:.0f}s)", flush=True)

s = rmsle(yev, np.expm1(np.clip(lp, 0, None)))
ens2s = np.mean([np.load(OUT / f"fx_se_2s_s{ss}.npy")
                 for ss in [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]], 0)
s2s = rmsle(yev, np.expm1(np.clip(ens2s, 0, None)))
corr = np.corrcoef(lp, ens2s)[0, 1]
print(f"kmix standalone {s:.5f}   two-stage ens9 {s2s:.5f}   corr {corr:.5f}")

ly = np.log1p(yev)
for label, vecs in [("2s + slope", [ens2s]), ("2s + kmix", [ens2s, lp])]:
    scores = []
    for r in range(3):
        half = np.random.default_rng(r).random(len(ly)) < 0.5
        A = np.stack(vecs + [np.ones_like(ly)], 1)
        w = np.linalg.lstsq(A[half], ly[half], rcond=None)[0]
        scores.append(rmsle(yev[~half], np.expm1(np.clip(A[~half] @ w, 0, None))))
    print(f"  {label:<12} held-out {np.mean(scores):.5f} +- {np.std(scores):.5f}")
np.save(OUT / "fx_kmix_s42.npy", lp.astype(np.float32))
