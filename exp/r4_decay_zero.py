"""Признаки затухания и структуры нулей — то, на что указали три независимые линии.

Проверка по solution_context: `decay`/`затухан` — 1 упоминание, то есть класса признаков нет.
А указали на него три разных источника независимо:
  * Favorita 1 место: `(x * 0.9^age).sum()` рядом с обычным средним того же окна, T1/2 ~ 6,6 дня;
  * Acquire Valued Shoppers, 43 место: `exp(-d/60)` — «единственный способ заставить gbm
    работать»;
  * Favorita 1 место и 6 место: признаки СТРУКТУРЫ нулей отдельными строками на каждое окно —
    число ненулевых дней, давность последнего ненулевого, давность первого ненулевого в окне,
    среднее ТОЛЬКО по ненулевым дням.

Почему это не то же, что наши оконные средние. Затухание не имеет резкой границы окна, поэтому
устойчиво к сдвигу якоря; а «среднее по ненулевым» и «давность первого ненулевого» вообще
ортогональны объёму — они про РЕЖИМ активности, а не про её величину.

Замер честный: тот же бустинг на тех же якорях, разница только в наборе колонок. Плюс допуск
в стенд, потому что нас интересует не только точность, но и вклад в стек.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import panel, target, rmsle, build_features, NU, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
CH = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
      "scart", "sord", "sday", "catday"]
HALF = [7, 21, 60, 180]
WINS = [7, 30, 90, 365]
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()

print("панели ...", flush=True)
P = {c: panel(c).astype(np.float32) for c in CH}

def block(anchor):
    """затухания и структура нулей по каждому каналу"""
    out, names = [], []
    for c in CH:
        M = P[c]
        for T in HALF:
            L = min(int(6 * T), anchor + 1)
            w = np.exp(-np.arange(L) * np.log(2) / T).astype(np.float32)
            seg = M[:, anchor - L + 1:anchor + 1]
            out.append(np.log1p(np.maximum((seg * w[::-1]).sum(1) / w.sum(), 0)))
            names.append(f"{c}_decay{T}")
        for W in WINS:
            L = min(W, anchor + 1)
            seg = M[:, anchor - L + 1:anchor + 1]
            nz = seg > 0
            cnt = nz.sum(1).astype(np.float32)
            idx = np.arange(L, dtype=np.float32)
            # давность последнего ненулевого: L-1 - максимальный индекс ненулевого
            last = np.where(cnt > 0, (L - 1) - (nz * idx).max(1), L)
            # давность первого ненулевого в окне
            first = np.where(cnt > 0, (L - 1) - np.where(nz, idx, L)[:, :].min(1), L)
            tot = seg.sum(1)
            mnz = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            out += [cnt / L, last / L, first / L, np.log1p(np.maximum(mnz, 0))]
            names += [f"{c}_nzcnt{W}", f"{c}_nzlast{W}", f"{c}_nzfirst{W}", f"{c}_nzmean{W}"]
    return np.stack(out, 1).astype(np.float32), names

print("строю базовый фичестор ...", flush=True)
X, base_names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
t0 = time.time()
NEW = {}
for a in sorted(set(SEL_TR + [SEL_EV])):
    NEW[a], new_names = block(a)
    print(f"  якорь {a}: +{NEW[a].shape[1]} колонок ({time.time()-t0:.0f}s)", flush=True)
print(f"базовых {len(base_names)}, новых {len(new_names)}", flush=True)

ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
PAR = dict(n_estimators=900, learning_rate=0.04, num_leaves=95, min_child_samples=150,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
           n_jobs=-1, verbose=-1)
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))

ARMS = {
    "базовый (165)":        (lambda a: X[a],                                  base_names),
    "только новые (220)":   (lambda a: NEW[a],                                new_names),
    "базовый + новые":      (lambda a: np.hstack([X[a], NEW[a]]),             base_names+new_names),
}
VEC = {}
print(f"\n{'рука':>20} {'колонок':>9} {'RMSLE':>10} {'сек':>7}", flush=True)
for nm, (get, nms) in ARMS.items():
    t0 = time.time()
    Xtr = np.concatenate([get(a) for a in SEL_TR], 0)
    m = lgb.LGBMRegressor(**PAR); m.fit(Xtr, ytr)
    v = m.predict(get(SEL_EV)).astype(np.float64)
    VEC[nm] = v
    np.save(OUT / f"r4_{'base' if 'базовый (' in nm else ('new' if 'только' in nm else 'both')}.npy",
            v.astype(np.float32))
    print(f"{nm:>20} {len(nms):9d} {sc(v):10.5f} {time.time()-t0:7.0f}", flush=True)
    if nm == "базовый + новые":
        imp = m.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        print("  топ-15 признаков объединённой руки:")
        for i in order:
            tag = "НОВЫЙ" if i >= len(base_names) else "     "
            print(f"    {tag} {nms[i]:>22} {imp[i]:8d}")
    del Xtr, m

print(f"\ncorr(базовый, базовый+новые) = {np.corrcoef(VEC['базовый (165)'], VEC['базовый + новые'])[0,1]:.5f}")
print(f"corr(базовый, только новые)  = {np.corrcoef(VEC['базовый (165)'], VEC['только новые (220)'])[0,1]:.5f}")

BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    if p.stem.startswith(("r1_","r2_","r4_","p8_","p9_")): continue
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (NU,) or not np.isfinite(v).all(): continue
    if sc(v.astype(np.float64)) < 1.80: BASE[p.stem] = v.astype(np.float64) - v.mean()
print(f"\nстенд: {len(BASE)} векторов")
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
print(f"\n{'кандидат':>20} {'A→B':>11} {'B→A':>11} {'среднее':>11} {'контроль':>11}")
for nm, v in VEC.items():
    vv = v - v.mean(); vp = rs.permutation(vv)
    d  = [fit_eval(cols+[vv],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    dc = [fit_eval(cols+[vp],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    print(f"{nm:>20} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} {np.mean(dc):+11.2e}")
print("\ndone")
