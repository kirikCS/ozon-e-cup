"""Recursive feature elimination — the version that actually works on correlated inputs.

Why not one-shot gain ranking (see feat_select.py, which found nothing): gain splits between
duplicated features. `gmv_30`, `gmv_60`, `gmv_90` and six block lags describe nearly the same
thing, so a tree spreads the credit and every member of the group looks mediocre. Rank once and
you cut blindly inside those groups. RFE refits after every removal, so the survivor's gain
rises the moment its twin is gone and the ranking stays honest.

Two things are measured at each step, not one:

  RMSLE          does the model itself get better
  corr to full   how far this model's predictions have drifted from the full-feature model

The second matters more for us. Component predictions currently correlate 0.992–0.999, which
makes the six-vector stack nearly degenerate; a pruned model that scores the same but
decorrelates is worth more to the blend than one that scores slightly better and doesn't.

Selection runs on its own anchor. VAL and HON are never touched here — confirm separately.

    python code/rfe.py                 # default schedule
    python code/rfe.py 0.15 40         # drop 15% per round, stop at 40 features
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from common import target, rmsle, build_features, ds, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
FLOOR = int(sys.argv[2]) if len(sys.argv) > 2 else 30

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")


def main():
    anchors = sorted(set(SEL_TR + [SEL_EV]))
    t0 = time.time()
    X, names = build_features(anchors)
    print(f"feature store: {len(names)} columns in {time.time()-t0:.0f}s")
    print(f"selection anchor {ds(SEL_EV)}, train {[ds(a) for a in SEL_TR]}\n")

    lytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    yev = target(SEL_EV)

    def fit(cols):
        m = lgb.LGBMRegressor(**LGB)
        m.fit(np.concatenate([X[a][:, cols] for a in SEL_TR], 0), lytr)
        lp = m.predict(X[SEL_EV][:, cols])
        return (rmsle(yev, np.expm1(lp)), lp,
                dict(zip(cols, m.booster_.feature_importance("gain"))))

    cols = list(range(len(names)))
    base_s, base_lp, imp = fit(cols)
    print(f"{'kept':>5} {'RMSLE':>9} {'vs base':>9} {'corr to full':>13}  {'dropped this round':<40} {'s':>4}")
    print(f"{len(cols):>5} {base_s:>9.5f} {0.0:>+9.5f} {1.0:>13.5f}  {'—':<40}")

    hist = [{"kept": len(cols), "rmsle": base_s, "corr_full": 1.0, "dropped": []}]
    best = {"kept": len(cols), "rmsle": base_s, "cols": list(cols)}
    while len(cols) > FLOOR:
        k = max(1, int(round(len(cols) * FRAC)))
        k = min(k, len(cols) - FLOOR)
        worst = sorted(cols, key=lambda c: imp.get(c, 0.0))[:k]
        cols = [c for c in cols if c not in set(worst)]
        t1 = time.time()
        s, lp, imp = fit(cols)
        corr = float(np.corrcoef(lp, base_lp)[0, 1])
        gone = ", ".join(names[c] for c in worst[:4]) + (f" +{k-4}" if k > 4 else "")
        flag = ""
        if s < best["rmsle"]:
            best = {"kept": len(cols), "rmsle": s, "cols": list(cols)}
            flag = "  <- best"
        print(f"{len(cols):>5} {s:>9.5f} {s-base_s:>+9.5f} {corr:>13.5f}  {gone:<40} {time.time()-t1:>4.0f}{flag}")
        hist.append({"kept": len(cols), "rmsle": s, "corr_full": corr,
                     "dropped": [names[c] for c in worst]})

    out = {"selection_anchor": ds(SEL_EV), "baseline_rmsle": base_s, "frac": FRAC, "floor": FLOOR,
           "history": hist, "best": {"kept": best["kept"], "rmsle": best["rmsle"],
                                     "features": [names[c] for c in best["cols"]]}}
    (OUT / "rfe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nbest single-model point: {best['kept']} features, RMSLE {best['rmsle']:.5f} "
          f"({best['rmsle']-base_s:+.5f})")
    print("decorrelation candidates (same score, lowest corr to the full model):")
    ok = [h for h in hist if h["rmsle"] <= base_s + 0.002]
    for h in sorted(ok, key=lambda d: d["corr_full"])[:5]:
        print(f"  {h['kept']:>4} features  RMSLE {h['rmsle']:.5f}  corr {h['corr_full']:.5f}")
    print(f"written to {OUT/'rfe.json'}")


if __name__ == "__main__":
    main()
