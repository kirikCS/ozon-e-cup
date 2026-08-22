"""Веер целевых функций на фичесторе — за ортогональностью, не за точностью.

Наша иерархия рычагов говорит, что смена функции потерь внутри одного пространства цели даёт
всего 0,9988 — как смена сида. Но здесь рычаг радикальнее: Tweedie и Пуассон моделируют E[y]
с ЛОГ-СВЯЗЬЮ на сырой цели, а не E[log1p y] в логарифмическом пространстве. Другое пространство,
другая связь, другая структура дисперсии.

Tweedie (компаунд Пуассон-Гамма) — стандарт для страховой чистой премии, а её структура в
точности наша: атом в нуле плюс непрерывный положительный хвост. При 1 < p < 2 распределение
имеет ровно такую смесь. Мы его не пробовали ни разу.

Как ПРЕДСКАЗАТЕЛЬ он заведомо смещён для RMSLE: log1p(E[y]) != E[log1p y], разрыв Йенсена
измерен и равен 1,0906. Поэтому смотрим не только на скор, но и на допуск в стек — гребневая
смесь наших векторов, веса на половине клиентов, скор на другой, контроль на перемешанном
кандидате.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
y = target(SEL_EV); ly = np.log1p(y)
print("фичестор ...", flush=True)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr_raw = np.concatenate([target(a) for a in SEL_TR], 0).astype(np.float64)
ytr_log = np.log1p(ytr_raw)
Xev = X[SEL_EV]
print(f"  {Xtr.shape}, доля нулей в обучении {(ytr_raw==0).mean():.4f}", flush=True)

COMMON = dict(n_estimators=700, learning_rate=0.05, num_leaves=95, min_child_samples=150,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              n_jobs=-1, verbose=-1)
ARMS = [
    ("l2_log1p",     dict(objective="regression"),                      "log"),
    ("l1_log1p",     dict(objective="regression_l1"),                   "log"),
    ("huber_log1p",  dict(objective="huber", alpha=1.0),                "log"),
    ("q50_log1p",    dict(objective="quantile", alpha=0.5),             "log"),
    ("q70_log1p",    dict(objective="quantile", alpha=0.7),             "log"),
    ("q90_log1p",    dict(objective="quantile", alpha=0.9),             "log"),
    ("poisson_raw",  dict(objective="poisson"),                         "raw"),
    ("tweedie_1.1",  dict(objective="tweedie", tweedie_variance_power=1.1), "raw"),
    ("tweedie_1.3",  dict(objective="tweedie", tweedie_variance_power=1.3), "raw"),
    ("tweedie_1.5",  dict(objective="tweedie", tweedie_variance_power=1.5), "raw"),
    ("tweedie_1.7",  dict(objective="tweedie", tweedie_variance_power=1.7), "raw"),
    ("tweedie_1.9",  dict(objective="tweedie", tweedie_variance_power=1.9), "raw"),
]

def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))

VEC = {}
print(f"\n{'рука':>14} {'простр.':>8} {'RMSLE':>10} {'сек':>7}", flush=True)
for nm, par, space in ARMS:
    t0 = time.time()
    m = lgb.LGBMRegressor(**COMMON, **par)
    m.fit(Xtr, ytr_log if space == "log" else ytr_raw)
    p = m.predict(Xev).astype(np.float64)
    v = p if space == "log" else np.log1p(np.maximum(p, 0))
    VEC[nm] = v
    np.save(OUT / f"r1_{nm}.npy", v.astype(np.float32))
    print(f"{nm:>14} {space:>8} {sc(v):10.5f} {time.time()-t0:7.0f}", flush=True)

print("\nкорреляции с базовой рукой l2_log1p:")
for nm in VEC:
    if nm != "l2_log1p":
        print(f"  {nm:>14}: {np.corrcoef(VEC[nm], VEC['l2_log1p'])[0,1]:.5f}")

# --- допуск в стек ---
BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    if p.stem.startswith(("r1_", "p8_")): continue
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (250000,) or not np.isfinite(v).all(): continue
    if sc(v.astype(np.float64)) < 1.80: BASE[p.stem] = v.astype(np.float64) - v.mean()
print(f"\nстенд: {len(BASE)} векторов")

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam*len(A)*np.eye(A.shape[1]); G[-1, -1] -= lam*len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

cols = list(BASE.values())
rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
SPL = ((half, ~half), (~half, half))
b = [fit_eval(cols, f, e) for f, e in SPL]
print(f"стенд без кандидатов: {np.mean(b):.6f}")
rs = np.random.default_rng(11)
print(f"\n{'кандидат':>14} {'A→B':>11} {'B→A':>11} {'среднее':>11} {'контроль':>11}")
gains = {}
for nm, v in VEC.items():
    vv = v - v.mean(); vp = rs.permutation(vv)
    d  = [fit_eval(cols + [vv], f, e) - b[i] for i, (f, e) in enumerate(SPL)]
    dc = [fit_eval(cols + [vp], f, e) - b[i] for i, (f, e) in enumerate(SPL)]
    gains[nm] = np.mean(d)
    print(f"{nm:>14} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} {np.mean(dc):+11.2e}",
          flush=True)
best = sorted(gains, key=gains.get)[:3]
allv = [VEC[n] - VEC[n].mean() for n in best]
d = [fit_eval(cols + allv, f, e) - b[i] for i, (f, e) in enumerate(SPL)]
print(f"\nлучшие три вместе ({', '.join(best)}): {np.mean(d):+.2e}")
print("\ndone")
