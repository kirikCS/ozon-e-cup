"""Расширенный фичестор на финальном якоре — продакшн.

r4: 220 новых признаков (затухания + структура нулей) дали 1,73847 в одиночку — почти столько
же, сколько весь наш фичестор из 165 колонок (1,73783). Вместе 1,73698. Вклад в стенд из 55
векторов −6,01e-04 при контроле +3,13e-06 — крупнейший из всех, что мы мерили.

Пять из пятнадцати важнейших признаков объединённой руки — новые: visit_decay7 (второй по
важности вообще), srch_nzmean365, catday_nzfirst365, srch_nzmean90, gmv_nzmean365.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import panel, target, rmsle, build_features, NU, OUT

SEL_TR, SEL_EV = [378, 350, 322, 294, 266], 408   # ФИНАЛЬНЫЙ протокол
CH = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
      "scart", "sord", "sday", "catday"]
HALF = [7, 21, 60, 180]
WINS = [7, 30, 90, 365]
y = None

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

from common import N_DAYS, H
assert SEL_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
PAR = dict(n_estimators=900, learning_rate=0.04, num_leaves=95, min_child_samples=150,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
           n_jobs=-1, verbose=-1)
acc = []
for sd in range(3):
    Xtr = np.concatenate([np.hstack([X[a], NEW[a]]) for a in SEL_TR], 0)
    m = lgb.LGBMRegressor(**PAR, random_state=sd); m.fit(Xtr, ytr)
    v = m.predict(np.hstack([X[SEL_EV], NEW[SEL_EV]])).astype(np.float64)
    acc.append(v)
    print(f"  сид {sd}: средняя log1p {v.mean():.5f}, sd {v.std():.5f}", flush=True)
    del Xtr, m
mm = np.mean(acc, 0)
np.save(OUT / "r7_boost_ext_fin.npy", mm.astype(np.float32))
print(f"\nусреднено 3 сида, corr(s0,s1) = {np.corrcoef(acc[0], acc[1])[0,1]:.5f}")
print("-> r7_boost_ext_fin.npy")
print("\ndone")
