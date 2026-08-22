"""Лестница порогов: E[log1p Y] как интеграл выживания. Тождество CREAD.

Ключевое тождество: для S = log1p(Y) >= 0

    E[S|x] = \\int_0^\\infty P(S > s|x) ds  ~=  sum_m h_m(x) * (s_m - s_{m-1})

где h_m(x) = P(log1p(Y) > s_m | x) — обычный бинарный классификатор. Это ТОЧНО тот функционал,
который оптимален под RMSLE: никакого распределительного допущения, никакой поправки Йенсена,
никакой симуляции путей. Нулевая масса берётся сама собой — нижняя перекладина это P(Y>0).

Почему это НЕ повтор закрытой 99-квантильной головы: та давала КВАНТИЛЬ, а лестница даёт
условное СРЕДНЕЕ log1p. Разные объекты. Квантиль был не тем функционалом.

Почему это НЕ повтор закрытой модели оттока: там вероятность использовалась МНОЖИТЕЛЕМ
(зануление, порог) и дала ноль. Здесь она — СЛАГАЕМОЕ интеграла. Математически другое
использование того же классификатора, отрицательный результат по первому на второе не переносится.

Почему это выдержит наши 250k: в формулировке «лестница» каждый классификатор видит ВСЕ примеры,
меняется только метка. В отличие от бакетных экспертов, где данные фрагментируются.

Два считывания дают два вектора почти бесплатно: среднее (интеграл) и медиана (где h пересекает
0,5) — второе заведомо сильнее декоррелировано.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
DELTA = float(__import__("os").environ.get("DELTA", 0.25))
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
print("фичестор ...", flush=True)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
Xev = X[SEL_EV]
TOP = float(np.quantile(ytr, 0.9995))
GRID = np.arange(DELTA, TOP, DELTA)
print(f"  {Xtr.shape}, порогов {len(GRID)}, шаг {DELTA}, верх {TOP:.2f}", flush=True)

P = np.zeros((len(y), len(GRID)), np.float32)
t0 = time.time()
for j, s in enumerate(GRID):
    lab = (ytr > s).astype(np.int8)
    fr = lab.mean()
    if fr < 1e-4 or fr > 1 - 1e-4:
        P[:, j] = fr; continue
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.07, num_leaves=63,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    m.fit(Xtr, lab)
    P[:, j] = m.predict_proba(Xev)[:, 1]
    if j % 8 == 0:
        print(f"  порог {s:5.2f}: доля {fr:.4f}, средняя p {P[:,j].mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
np.save(OUT / "r2_ladder_P.npy", P)

def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))
print(f"\nкалибровка лестницы (средняя p против факта):", flush=True)
for j in range(0, len(GRID), 10):
    print(f"  s={GRID[j]:5.2f}: модель {P[:,j].mean():.4f}, факт {(ly>GRID[j]).mean():.4f}")

mean_ro = DELTA * P.sum(1)                                   # интеграл выживания
Pm = np.maximum.accumulate(P[:, ::-1], 1)[:, ::-1]           # монотонизация для медианы
med_ro = np.where((Pm > 0.5).any(1), GRID[np.argmin(Pm > 0.5, 1)], 0.0)
print(f"\n{'считывание':>14} {'RMSLE':>10}")
for nm, v in (("интеграл", mean_ro), ("медиана", med_ro)):
    print(f"{nm:>14} {sc(v):10.5f}")
    np.save(OUT / f"r2_ladder_{'mean' if nm=='интеграл' else 'median'}.npy", v.astype(np.float32))
print(f"corr(интеграл, медиана) = {np.corrcoef(mean_ro, med_ro)[0,1]:.5f}")

BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    if p.stem.startswith(("r1_", "r2_", "p8_", "p9_")): continue
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (len(y),) or not np.isfinite(v).all(): continue
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
for nm, v in (("интеграл", mean_ro), ("медиана", med_ro)):
    vv = v - v.mean(); vp = rs.permutation(vv)
    d  = [fit_eval(cols + [vv], f, e) - b[i] for i, (f, e) in enumerate(SPL)]
    dc = [fit_eval(cols + [vp], f, e) - b[i] for i, (f, e) in enumerate(SPL)]
    print(f"{nm:>14} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} {np.mean(dc):+11.2e}")
d = [fit_eval(cols + [mean_ro-mean_ro.mean(), med_ro-med_ro.mean()], f, e) - b[i]
     for i, (f, e) in enumerate(SPL)]
print(f"{'обе вместе':>14} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e}")
print("\ndone")
