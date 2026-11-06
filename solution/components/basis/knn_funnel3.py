"""knn_funnel3: 5-d "trend + complete-chain" kNN.

Features NOT used in knn_funnel1/2:
  srch_share_shift_30v90   -- search share CHANGE (30d vs 90d)
  srch_share_blk1v3_trend  -- early vs. late search block dynamics
  buyday_freq_trend         -- buying frequency trend
  triple_cooccur_30         -- complete srch->cart->ord chain (30d)
  triple_cooccur_90         -- complete srch->cart->ord chain (90d)

These capture HOW BEHAVIOR IS CHANGING and COMPLETE CONVERSION,
orthogonal to the static rates in f1 (srch_cart, cart_ord, ord_per_srch)
and f2 (same + volume). Expected lower corr(ev3) → better decorrelation.

K=200, sigma=10, CPU FAISS IndexFlatIP, features L2-normalized.

Usage: python -u code/knn_funnel3.py > model_out/knn_funnel3.log 2>&1
"""
import sys, time
import numpy as np
import polars as pl
import faiss
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import NU, target, rmsle, OUT
from extra_features import build_extra_features

EL    = 2.3284
CACHE = Path('cache')
uids  = np.load(CACHE / 'uids.npy')
EV    = 348
LB_ANCHOR = 408

TR_POOL = [318, 290, 262, 234, 206]
LB_POOL = [378, 348, 320, 292, 264]
K_NN    = 200
SIGMA   = 10.0

# 5-d trend + complete-chain features
TREND_FEATS = [
    "srch_share_shift_30v90",
    "srch_share_blk1v3_trend",
    "buyday_freq_trend",
    "triple_cooccur_30",
    "triple_cooccur_90",
]

all_anchors = sorted(set(TR_POOL + LB_POOL + [EV, LB_ANCHOR]))
print(f"Building extra features for {len(all_anchors)} anchors...", flush=True)
t0 = time.time()
X_ext, ext_names = build_extra_features(all_anchors, verbose=False)
fidx = [ext_names.index(f) for f in TREND_FEATS]
print(f"  Done ({time.time()-t0:.0f}s). Feature indices: {fidx}", flush=True)


def get_feat(anchor):
    F = X_ext[anchor][:, fidx].astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # L2-normalize each feature column to unit std (handle buyday_freq_trend's high variance)
    col_std = F.std(0)
    col_std[col_std < 1e-8] = 1.0
    return F / col_std[None, :]


def knn_predict(X_qry, anchors_train, y_train):
    """kNN in feature space with Gaussian kernel, CPU FAISS."""
    X_tr_list = [get_feat(a) for a in anchors_train]
    y_tr_list  = [np.log1p(target(a).astype(np.float64)) for a in anchors_train]
    X_tr = np.vstack(X_tr_list).astype(np.float32)
    y_tr = np.concatenate(y_tr_list).astype(np.float64)

    # FAISS CPU exact L2 search
    d = X_tr.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(X_tr)
    print(f"  FAISS index: {index.ntotal} train pts, {X_qry.shape[0]} queries", flush=True)

    t1 = time.time()
    D, I = index.search(X_qry.astype(np.float32), K_NN)
    print(f"  kNN search done ({time.time()-t1:.0f}s)", flush=True)

    # Gaussian kernel weights
    W = np.exp(-D / (2 * SIGMA**2))
    W_sum = W.sum(1, keepdims=True)
    W_sum[W_sum < 1e-30] = 1.0
    W /= W_sum

    pred = (W * y_tr[I]).sum(1)
    return pred


print("\n--- LOCAL eval ---", flush=True)
X_ev = get_feat(EV)
y_ev = np.log1p(target(EV).astype(np.float64))
p_local = knn_predict(X_ev, TR_POOL, None)
p_cal   = p_local - p_local.mean() + EL
sc_local = rmsle(target(EV), np.expm1(np.clip(p_cal, 0, 13)))
print(f"LOCAL={sc_local:.5f}", flush=True)

# Correlations with existing funnel models
try:
    ev3 = np.mean([np.load(str(OUT/f'fx_evgru3_s{s}.npy')) for s in [0,1]
                   if (OUT/f'fx_evgru3_s{s}.npy').exists()], 0)
    print(f"corr(ev3)={np.corrcoef(p_cal, ev3)[0,1]:.5f}", flush=True)
except Exception as e:
    print(f"corr(ev3) unavailable: {e}", flush=True)

try:
    fl = np.load(str(OUT/'fx_funnel_lgbm_local.npy'))
    print(f"corr(funnel_lgbm)={np.corrcoef(p_cal, fl)[0,1]:.5f}", flush=True)
except Exception: pass

# Expected LB (LOCAL * 0.963 ratio)
exp_lb = sc_local * 0.963
print(f"Expected LB ≈ {exp_lb:.4f}", flush=True)
np.save(str(OUT/'fx_knn_funnel3_local.npy'), p_cal.astype(np.float32))

print("\n--- LB submission ---", flush=True)
X_lb = get_feat(LB_ANCHOR)
p_lb = knn_predict(X_lb, LB_POOL, None)
p_lb_cal = p_lb - p_lb.mean() + EL
pv_lb = np.clip(np.expm1(p_lb_cal), 0, None)

# Cross-corr with existing funnel LB vectors
try:
    d2 = pl.read_csv('submissions/knn_funnel2_lb.csv')
    o2 = np.argsort(d2['user_id'].to_numpy())
    ev2_lb = np.log1p(d2['predict'].to_numpy()[o2])
    print(f"corr(knn_funnel2)={np.corrcoef(p_lb_cal, ev2_lb)[0,1]:.5f}", flush=True)
except Exception: pass

try:
    d3 = pl.read_csv('submissions/evgru3_final_lb1656201.csv')
    o3 = np.argsort(d3['user_id'].to_numpy())
    ev3_lb = np.log1p(d3['predict'].to_numpy()[o3])
    print(f"corr(ev3_lb)={np.corrcoef(p_lb_cal, ev3_lb)[0,1]:.5f}", flush=True)
except Exception: pass

out = Path('submissions/knn_funnel3_lb.csv')
with open(out, 'w') as f:
    f.write('user_id,predict\n')
    for u, v in zip(uids, pv_lb):
        f.write(f'{u},{v:.6f}\n')
print(f"Wrote {out}")
print(f"n={len(uids)}, mean_log1p={np.log1p(pv_lb+1e-9).mean():.4f}, "
      f"min={pv_lb.min():.4f}, max={pv_lb.max():.1f}")
print("Done.")
