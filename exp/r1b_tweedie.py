"""Руки Tweedie и итоговый допуск веера целевых функций.

r1 сняли на седьмой руке из-за конкуренции за ядра с лестницей; семь векторов сохранены.
Здесь досчитываются пять рук Tweedie — ради них веер и затевался: прецедент M5, где первое,
третье и четвёртое места учили LightGBM на правдоподобии Tweedie, а пятое на пуассоне.

Оговорка, которую надо держать в голове: метрика M5 на СЫРОМ масштабе, где E[Y] — правильная
цель. У нас метрика логарифмическая, поэтому Tweedie на сыром y заведомо смещён, и ценность
только в ортогональности. Разбор дал и теоретическую рамку: девианс любого EDM, применённый
к z = log1p(y), состоятелен для E[z|x] при ЛЮБОМ p — выбор p это вопрос весов, не смещения.
Поэтому здесь обе версии: Tweedie на сыром y и Tweedie на log1p.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import target, rmsle, build_features, NU, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
print("фичестор ...", flush=True)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr_raw = np.concatenate([target(a) for a in SEL_TR], 0).astype(np.float64)
ytr_log = np.log1p(ytr_raw)
Xev = X[SEL_EV]
COMMON = dict(n_estimators=700, learning_rate=0.05, num_leaves=95, min_child_samples=150,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              n_jobs=-1, verbose=-1)
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))

ARMS = [(f"tweedie_{p}_raw", dict(objective="tweedie", tweedie_variance_power=p), "raw")
        for p in (1.1, 1.3, 1.5, 1.7, 1.9)] + \
       [(f"tweedie_{p}_log", dict(objective="tweedie", tweedie_variance_power=p), "log")
        for p in (1.1, 1.5, 1.9)]
print(f"\n{'рука':>18} {'простр.':>8} {'RMSLE':>10} {'сек':>7}", flush=True)
for nm, par, space in ARMS:
    t0 = time.time()
    m = lgb.LGBMRegressor(**COMMON, **par)
    m.fit(Xtr, ytr_log if space == "log" else ytr_raw)
    p = m.predict(Xev).astype(np.float64)
    v = p if space == "log" else np.log1p(np.maximum(p, 0))
    np.save(OUT / f"r1_{nm}.npy", v.astype(np.float32))
    print(f"{nm:>18} {space:>8} {sc(v):10.5f} {time.time()-t0:7.0f}", flush=True)
    del m

VEC = {}
for p in sorted(Path(OUT).glob("r1_*.npy")):
    v = np.load(p).astype(np.float64)
    if v.shape == (NU,) and np.isfinite(v).all():
        VEC[p.stem[3:]] = v
print(f"\nвсего рук веера: {len(VEC)}", flush=True)
base = VEC.get("l2_log1p")
if base is not None:
    print("корреляции с базовой l2_log1p:")
    for nm in sorted(VEC, key=lambda k: sc(VEC[k])):
        print(f"  {nm:>18} {sc(VEC[nm]):9.5f}  corr {np.corrcoef(VEC[nm], base)[0,1]:.5f}")

BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    if p.stem.startswith(("r1_","r2_","r4_","r5_","r6_","r7_","r8_","r9_","p8_","p9_")): continue
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (NU,) or not np.isfinite(v).all(): continue
    if sc(v.astype(np.float64)) < 1.80: BASE[p.stem] = v.astype(np.float64) - v.mean()
print(f"\nстенд: {len(BASE)} векторов (лестница и фичестор ИСКЛЮЧЕНЫ)")
def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(),1))], 1)
    G = A.T@A + lam*len(A)*np.eye(A.shape[1]); G[-1,-1] -= lam*len(A)
    w = np.linalg.solve(G, A.T@ly[fm])
    v = M@w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em],0,13)))
cols = list(BASE.values())
rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
SPL = ((half,~half),(~half,half))
b = [fit_eval(cols,f,e) for f,e in SPL]
print(f"стенд без кандидатов: {np.mean(b):.6f}")
rs = np.random.default_rng(11)
print(f"\n{'кандидат':>18} {'среднее':>11} {'контроль':>11} {'на стеке':>11}")
res = {}
for nm in sorted(VEC, key=lambda k: sc(VEC[k])):
    vv = VEC[nm] - VEC[nm].mean(); vp = rs.permutation(vv)
    d  = [fit_eval(cols+[vv],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    dc = [fit_eval(cols+[vp],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    res[nm] = np.mean(d)
    print(f"{nm:>18} {np.mean(d):+11.2e} {np.mean(dc):+11.2e} {np.mean(d)/7.8:+11.2e}")
best = sorted(res, key=res.get)[:4]
allv = [VEC[n]-VEC[n].mean() for n in best]
d = [fit_eval(cols+allv,f,e)-b[i] for i,(f,e) in enumerate(SPL)]
print(f"\nлучшие четыре вместе ({', '.join(best)}): {np.mean(d):+.2e} "
      f"-> на стеке {np.mean(d)/7.8:+.2e}")
print("\ndone")
