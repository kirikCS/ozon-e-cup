"""Adversarial validation: can a classifier tell SELECT-training rows from anchor-408 rows?

Self-closing diagnostic from the literature sweep. AUC <= ~0.55 -> no exploitable covariate
shift between the training anchors and the real prediction anchor, and the reweighting idea
closes without a model run. High AUC -> the top drift features name what actually moves.

The four anchor-ID columns (tgt_doy_sin/cos, tgt_month, hist_days) are constant within an
anchor and would separate trivially — excluded up front; the interesting drift is in the
behavioural columns.

Usage:  python -u code/fx_advval.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, build_features

TR, PRED = [318, 290, 262, 234, 206], 408

X, names = build_features(sorted(set(TR + [PRED])))
drop = {"tgt_doy_sin", "tgt_doy_cos", "tgt_month", "hist_days"}
keep = [i for i, n in enumerate(names) if n not in drop]
kn = [names[i] for i in keep]

rng = np.random.default_rng(0)
# balanced: 50k random rows from each training anchor (250k total) vs the 250k rows at 408
parts, labels = [], []
for a in TR:
    idx = rng.choice(NU, 50_000, replace=False)
    parts.append(X[a][idx][:, keep]); labels.append(np.zeros(len(idx), np.int8))
parts.append(X[PRED][:, keep]); labels.append(np.ones(NU, np.int8))
Xa = np.concatenate(parts, 0); ya = np.concatenate(labels)

half = rng.random(len(ya)) < 0.5
t0 = time.time()
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63,
                       min_child_samples=100, max_bin=63, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.7, reg_lambda=5.0, n_jobs=23, verbose=-1,
                       random_state=42, force_col_wise=True)
m.fit(Xa[half], ya[half])
auc = roc_auc_score(ya[~half], m.predict_proba(Xa[~half])[:, 1])
print(f"adversarial AUC (train anchors vs 408): {auc:.4f}   ({time.time()-t0:.0f}s)")

imp = m.booster_.feature_importance("gain")
top = np.argsort(imp)[::-1][:15]
tot = imp.sum()
print("top drift features:")
for i in top:
    print(f"  {kn[i]:<24s} {100*imp[i]/tot:5.2f}%")
