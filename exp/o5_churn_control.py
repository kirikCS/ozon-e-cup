"""Специфичен ли выигрыш для оттока — или это просто десять свободных параметров.

В o4 подецильные сдвиги по p_отток дали -2,8e-4 поверх бустинга. Два изъяна постановки:
  1. опора собиралась равновесной смесью БЕЗ подбора весов; неоптимальные веса сами создают
     условное смещение, и сдвиги чинят его, а не отток;
  2. не было контроля на «десять свободных параметров».

Здесь оба закрыты. Веса смеси подгоняются гребнем на той же половине A, что и поправка. Контроли:
  * сдвиги по децилям САМОГО ПРЕДСКАЗАНИЯ — обычная перекалибровка, оттока не содержит;
  * сдвиги по перемешанной p_отток — чистый нуль на те же десять параметров;
  * сдвиги по децилям СЛУЧАЙНОГО признака — второй нуль.
Выигрыш засчитывается только сверх лучшего из контролей.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
y = target(SEL_EV); ly = np.log1p(y)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)

def fit_lgb(obj):
    Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
    if obj == "reg":
        ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
        m = lgb.LGBMRegressor(n_estimators=900, learning_rate=0.04, num_leaves=95,
                              min_child_samples=150, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    else:
        ytr = np.concatenate([(target(a) > 0).astype(np.int8) for a in SEL_TR], 0)
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr); return m

t0 = time.time()
v_lgb = fit_lgb("reg").predict(X[SEL_EV]).astype(np.float64)
pch   = 1.0 - fit_lgb("clf").predict_proba(X[SEL_EV])[:, 1]
print(f"бустинг и классификатор готовы ({time.time()-t0:.0f}s), "
      f"AUC {roc_auc_score((y>0).astype(int), 1-pch):.5f}", flush=True)

NEUR = ["h8_gauss", "d7_fusion4_sel", "d6_fusion_sel", "c5_sel", "d2_fusion4_sel",
        "g3_all_sel", "h2_concat_bag_sel"]
COMP = [np.load(OUT / f"{n}.npy") for n in NEUR if (OUT / f"{n}.npy").exists()]
COMP.append(v_lgb)
M = np.stack([c - c.mean() for c in COMP], 1)
print(f"компонент в смеси: {M.shape[1]} (в т.ч. бустинг)", flush=True)

def blend(fm, lam=1e-4):
    """веса гребнем на половине A — так же, как это делает стек"""
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam*len(A)*np.eye(A.shape[1]); G[-1, -1] -= lam*len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    return M @ w[:-1] + w[-1]

def sc(v, m): return rmsle(y[m], np.expm1(np.clip(v[m], 0, 13)))

def decile_shift(vb, key, fm, nb=10):
    q = np.quantile(key, np.linspace(0, 1, nb+1)); q[0] -= 1e-9; q[-1] += 1e-9
    d = np.clip(np.digitize(key, q[1:-1]), 0, nb-1)
    w = vb.copy()
    for k in range(nb):
        m = d == k
        if (m & fm).sum() > 50:
            w[m] += (ly[m & fm] - vb[m & fm]).mean()
    return w

rs = np.random.default_rng(7)
KEYS = {
    "p_отток":                    pch,
    "КОНТРОЛЬ: само предсказание": None,          # подставится vb
    "КОНТРОЛЬ: p_отток перемешан": rs.permutation(pch),
    "КОНТРОЛЬ: случайный признак": rs.standard_normal(len(y)),
}
rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
print(f"\n{'ключ децилей':>30} {'A→B':>11} {'B→A':>11} {'среднее':>11}")
base_scores = []
res = {}
for kname, key in KEYS.items():
    d = []
    for fm, em in ((half, ~half), (~half, half)):
        vb = blend(fm)
        s0 = sc(vb, em)
        if len(base_scores) < 2: base_scores.append(s0)
        kk = vb if key is None else key
        d.append(sc(decile_shift(vb, kk, fm), em) - s0)
    res[kname] = float(np.mean(d))
    print(f"{kname:>30} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e}", flush=True)
print(f"\nопора (гребневая смесь): {base_scores[0]:.5f} / {base_scores[1]:.5f}")

ctrl = min(res[k] for k in res if k.startswith("КОНТРОЛЬ"))
print(f"\nлучший контроль: {ctrl:+.2e}")
print(f"выигрыш p_отток СВЕРХ лучшего контроля: {res['p_отток'] - ctrl:+.2e}")

# и сверх обычной перекалибровки: двумерно, отток внутри дециля предсказания
print("\nдвумерно: сначала перекалибровка по предсказанию, потом отток внутри")
d = []
for fm, em in ((half, ~half), (~half, half)):
    vb = blend(fm); s0 = sc(vb, em)
    v1 = decile_shift(vb, vb, fm)
    v2 = decile_shift(v1, pch, fm)
    d.append((sc(v1, em) - s0, sc(v2, em) - s0))
print(f"  только перекалибровка: {np.mean([x[0] for x in d]):+.2e}")
print(f"  + отток сверху:        {np.mean([x[1] for x in d]):+.2e}")
print(f"  чистый вклад оттока:   {np.mean([x[1]-x[0] for x in d]):+.2e}")
print("\ndone")
