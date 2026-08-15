"""Is the Catalog/Search split a stable property of a client, or noise?

The question decides what the split is worth. If a client's monthly catalog share is a personal
constant, it is a type indicator worth carrying as a feature. If it varies month to month as
much as it varies between clients, there is no constant to estimate and the split only adds
variance.

That is exactly what the intraclass correlation measures:

    ICC = var_between_clients / (var_between_clients + var_within_client)

computed on monthly catalog shares, over clients with enough orders for the share to mean
anything. ICC near 1 — a personal constant. ICC near 0 — nothing to estimate.

Reported alongside:
  - the population share by month, to separate a seasonal drift from client heterogeneity
  - the reliability of the per-client estimate as a function of how many orders it rests on
    (Spearman-Brown: an average of k noisy readings has reliability k*ICC/(1+(k-1)*ICC))
  - and then the practical test: the shrunken per-client constant plus its dispersion and
    trend, added to the production LightGBM, three seeds, against the noise band.

gmv = gmv_search + gmv_cat holds exactly on all 30.6M rows, so gmv_cat needs no panel of its
own; the catalog side is 7.4% of GMV mass and fires on 1.08% of rows, which is why nothing in
the feature store has ever seen it separately.
"""
import sys

import numpy as np
import lightgbm as lgb

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, ds

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
SEEDS = (42, 101, 777)
LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=11, verbose=-1, force_col_wise=True, objective="regression")

G, GS = panel("gmv"), panel("gmv_search")
GC = G - GS
ORD = panel("ord")
try:
    SORD = panel("sord")
    CORD = np.maximum(ORD - SORD, 0)
except Exception:
    CORD = None

NB = N_DAYS // 30
print("=== 1. месячная доля каталога: сезонность против личных различий ===")
blocks = [(N_DAYS - 30 * (k + 1), N_DAYS - 30 * k - 1) for k in range(NB)][::-1]
gc_m = np.stack([GC[:, s:e + 1].sum(1) for s, e in blocks], 1)
g_m = np.stack([G[:, s:e + 1].sum(1) for s, e in blocks], 1)
share = np.where(g_m > 0, gc_m / np.maximum(g_m, 1e-9), np.nan)
print(f"{'блок':>6} {'дней':>6} {'доля каталога по популяции':>28} {'клиентов с покупкой':>21}")
for i, (s, e) in enumerate(blocks):
    m = g_m[:, i] > 0
    print(f"{i:6d} {e-s+1:6d} {gc_m[:,i].sum()/max(g_m[:,i].sum(),1e-9):28.4f} {int(m.sum()):21d}")

print("\n=== 2. ICC: доля каталога — личная константа или шум ===")
for min_months in (3, 5, 8):
    ok = (~np.isnan(share)).sum(1) >= min_months
    S = share[ok]
    n_i = (~np.isnan(S)).sum(1)
    mu_i = np.nanmean(S, 1)
    grand = np.nansum(S) / np.nansum(~np.isnan(S))
    within = np.nansum((S - mu_i[:, None]) ** 2) / max(np.nansum(~np.isnan(S)) - len(S), 1)
    k0 = n_i.mean()
    between = (np.sum(n_i * (mu_i - grand) ** 2) / max(len(S) - 1, 1) - within) / k0
    between = max(between, 0.0)
    icc = between / max(between + within, 1e-12)
    sb = k0 * icc / (1 + (k0 - 1) * icc)
    print(f"  >= {min_months} месяцев с покупками: {len(S):6d} клиентов, "
          f"ICC {icc:.4f}, надёжность среднего по {k0:.1f} месяцам {sb:.4f}")

print("\n=== 3. признаки из канала, поверх продакшн-LightGBM ===")
def chan_feats(a):
    """shrunk personal catalog share plus its dispersion and drift, history only"""
    bl = [(max(0, a - 30 * (k + 1) + 1), a - 30 * k) for k in range(6)]
    gcm = np.stack([GC[:, s:e + 1].sum(1) for s, e in bl], 1)
    gm = np.stack([G[:, s:e + 1].sum(1) for s, e in bl], 1)
    tot_c, tot_g = gcm.sum(1), gm.sum(1)
    pop = tot_c.sum() / max(tot_g.sum(), 1e-9)
    # эмпирический байес: приор — популяционная доля, вес приора 3 заказа
    ordc = ORD[:, max(0, a - 179):a + 1].sum(1)
    sh_shrunk = (tot_c + 3 * pop * np.maximum(tot_g, 1e-9) / np.maximum(ordc, 1)) \
        / np.maximum(tot_g + 3 * np.maximum(tot_g, 1e-9) / np.maximum(ordc, 1), 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(gm > 0, gcm / np.maximum(gm, 1e-9), np.nan)
    disp = np.nan_to_num(np.nanstd(per, 1), nan=0.0)
    nmon = np.nan_to_num((~np.isnan(per)).sum(1), nan=0.0)
    recent = np.nan_to_num(np.nanmean(per[:, :2], 1), nan=pop)
    older = np.nan_to_num(np.nanmean(per[:, 3:], 1), nan=pop)
    cols = [np.clip(sh_shrunk, 0, 1), disp, nmon, recent - older,
            np.log1p(tot_c), (tot_c > 0).astype(np.float64)]
    if CORD is not None:
        co = CORD[:, max(0, a - 179):a + 1].sum(1)
        cols += [np.log1p(co), co / np.maximum(ordc, 1)]
    return np.stack(cols, 1).astype(np.float32)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
yev = target(SEL_EV)
for tag, use in (("база", False), ("+признаки канала", True)):
    Xtr = np.concatenate([np.concatenate([X[a], chan_feats(a)], 1) if use else X[a]
                          for a in SEL_TR], 0)
    Xev = np.concatenate([X[SEL_EV], chan_feats(SEL_EV)], 1) if use else X[SEL_EV]
    s = [rmsle(yev, np.expm1(lgb.LGBMRegressor(**dict(LGB, random_state=sd))
                             .fit(Xtr, ytr).predict(Xev))) for sd in SEEDS]
    print(f"  {tag:20s} {np.mean(s):.5f}   сиды " + ", ".join(f"{v:.5f}" for v in s))
    del Xtr
print("\ndone")
