"""Re-run the linear baseline properly.

The first attempt used alpha=10 on 165 heavily collinear log-features; sklearn reported an
ill-conditioned matrix and the HON protocol produced a meaningless 27.4. Fixed by dropping
near-constant columns, standardising on the training statistics only, and regularising properly.
"""
import pickle
import sys
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, NU, target, cohort_rule, rmsle, record, Timer,
                    build_features, OUT)

A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264, 236]
TR_HON = [258, 230, 202, 174, 146]
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))
X, NAMES = build_features(ALL_A)
try:
    BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
    # only attach BTYD columns if every anchor has them, otherwise the blocks differ in width
    if all(a in BT for a in ALL_A):
        for a in ALL_A:
            X[a] = np.concatenate([X[a], BT[a]], 1)
    else:
        print("  BTYD features missing for", [a for a in ALL_A if a not in BT], "- skipping them")
except Exception as e:
    print("  no BTYD features:", e)
Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)
PROTO = {"VAL": (TR_VAL, A, None), "HON": (TR_HON, AH, m_hon)}

def run(alpha, name):
    res, secs = {}, 0
    for key, (tr, ev, mask) in PROTO.items():
        with Timer() as t:
            Z = np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in tr], 0)
            Z = np.nan_to_num(Z, posinf=0, neginf=0)
            sd = Z.std(0)
            keep = sd > 1e-3                     # drop columns with no variation in this protocol
            mu, sd = Z[:, keep].mean(0), sd[keep]
            Ztr = np.clip((Z[:, keep] - mu) / sd, -8, 8)
            ly = np.concatenate([np.log1p(Y[a]) for a in tr])
            m = Ridge(alpha=alpha)
            m.fit(Ztr, ly)
            Ze = np.log1p(np.clip(X[ev], 0, None))
            Ze = np.nan_to_num(Ze, posinf=0, neginf=0)[:, keep]
            p = np.expm1(np.clip(m.predict(np.clip((Ze - mu) / sd, -8, 8)), 0, 13))
        secs += t.dt
        res[key] = rmsle(Y[ev], p) if mask is None else rmsle(Y[ev][mask], p[mask])
        if key == "VAL":
            np.save(OUT / f"pred_{name.replace(' ', '_')[:60]}.npy", p.astype(np.float32))
    return record(name, "linear", res["VAL"], res["HON"], secs,
                  f"alpha={alpha}, near-constant columns dropped, standardised on train")

print("--- Tier 3b: linear baseline, fixed ---")
for al in [100.0, 1000.0]:
    run(al, f"Ridge on log-features -> log1p target (alpha={int(al)})")
print("done")
