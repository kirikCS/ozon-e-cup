"""TabPFN v2 and the TabPFN-TS formulation.

TabPFN is a prior-fitted network: it does no gradient training, it conditions in-context on a
support set and predicts the query in one forward pass. Two constraints shape the implementation:

  * the context is limited to roughly ten thousand rows, while we have a million — so the support
    set is subsampled and the whole thing is bagged over several draws and averaged in log space;
  * inference cost is O(context x query), and the query is 250k rows, so it is chunked.

TabPFN-TS is not a different network. It is TabPFN applied to a tabular matrix that carries
temporal features — lags, windows, calendar position — which is exactly the matrix we already
build. The second job below makes that explicit by restricting the columns to the temporal block,
so the difference between the two rows is the feature view, not the model.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "models")
from zoo_common import load, emit, TR_PROBE, TR_FINAL, FIT, PRED
from common import rmsle, ds

X, Y, NAMES, TOP = load()
NF = 64                     # TabPFN degrades on very wide inputs; take the strongest columns
CTX = 8192                  # support-set size per bag
BAGS = 6
QCHUNK = 8192

TEMPORAL = [i for i, n in enumerate(NAMES) if any(
    n.startswith(p) for p in ("gmv_", "ord_", "cart_", "srch_", "visit_", "engaged_", "buyday_",
                              "wk_", "slope_", "rec_", "tgt_", "hist_", "bgap_", "vgap_",
                              "tsb_", "sba_", "eb_", "btyd_"))]
COLS_ALL = np.sort(TOP[:NF])
COLS_TS = np.sort(np.array([c for c in TOP if c in set(TEMPORAL)][:NF]))
print(f"TabPFN: {len(COLS_ALL)} колонок (top по gain), TabPFN-TS: {len(COLS_TS)} временных")

def tabpfn_fit(cols):
    def f(Xtr, ytr, Xev):
        from tabpfn import TabPFNRegressor
        ly = np.log1p(ytr).astype(np.float32)
        rng = np.random.default_rng(0)
        acc = np.zeros(len(Xev), dtype=np.float64)
        for b in range(BAGS):
            idx = rng.choice(len(Xtr), min(CTX, len(Xtr)), replace=False)
            m = TabPFNRegressor(device="cuda", n_estimators=2, random_state=b)
            m.fit(Xtr[idx][:, cols], ly[idx])
            out = np.empty(len(Xev), dtype=np.float64)
            for i in range(0, len(Xev), QCHUNK):
                out[i:i + QCHUNK] = m.predict(Xev[i:i + QCHUNK][:, cols])
            acc += out
            del m
        return acc / BAGS
    return f

JOBS = [
    ("TabPFN v2 (bagged in-context)", "zoo_tabpfn", COLS_ALL),
    ("TabPFN-TS (временной срез признаков)", "zoo_tabpfn_ts", COLS_TS),
]
for name, fname, cols in JOBS:
    print(f"\n{name}")
    t0 = time.time()
    try:
        fn = tabpfn_fit(np.arange(len(cols)))     # columns are pre-sliced below
        Xtr = np.concatenate([X[a][:, cols] for a in TR_PROBE], 0)
        ytr = np.concatenate([Y[a] for a in TR_PROBE])
        lp = fn(Xtr, ytr, X[FIT][:, cols])
        del Xtr
        print(f"    probe {ds(FIT)}: {rmsle(Y[FIT], np.expm1(np.clip(np.nan_to_num(lp),0,13))):.5f}"
              f"  ({time.time()-t0:.0f}s)")
        Xtr = np.concatenate([X[a][:, cols] for a in TR_FINAL], 0)
        ytr = np.concatenate([Y[a] for a in TR_FINAL])
        emit(fn(Xtr, ytr, X[PRED][:, cols]), fname)
        del Xtr
    except Exception as e:
        import traceback; print(f"    FAILED: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\ndone")
