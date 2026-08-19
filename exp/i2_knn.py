"""kNN-регрессия как отдельное семейство — инстансная модель против параметрических.

Зачем. Стек платит только за ортогональность: смена функции потерь двигает вектор на 0.9988,
смена архитектуры на 0.993-0.996, смена семейства представления (событийное против календарного)
на 0.988. Все крупные оси представления перебраны. Осталась ось, которую мы не трогали ни разу —
тип модели: всё в стеке параметрическое (деревья, свёртки, attention, рекуррентные). Инстансная
модель не обучает параметров вообще, она усредняет соседей, и ошибается там, где соседей мало,
а не там, где функция плохо приближена.

kNN уже применялся в проекте, но как ОЦЕНЩИК ПОЛА (s1: 1.6409 на 20 признаках), а не как модель:
там соседи брались на том же якоре с известным таргетом. Здесь честно — соседи ищутся среди
обучающих якорей, таргет берётся их.

Руки:
  knn_k50/k200/k500   разное сглаживание на стандартизованных top-признаках
  knn_pca             сначала PCA до 24 компонент — метрика в пространстве без дублей
  knn_rank            признаки переведены в ранги — метрика устойчива к хвостам

Смотрим на corr с дневной моделью. Скор заведомо будет хуже: вопрос только в том, насколько
далеко инстансная ошибка лежит от параметрической.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, OUT
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
NTOP = 24

X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
mu = np.mean([X[a].mean(0) for a in SEL_TR], 0)
sd = np.maximum(np.mean([X[a].std(0) for a in SEL_TR], 0), 1e-2)

import lightgbm as lgb
Xtr_full = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=100,
                      max_bin=63, n_jobs=11, verbose=-1, random_state=42,
                      force_col_wise=True).fit(Xtr_full, ytr)
top = np.argsort(m.booster_.feature_importance("gain"))[::-1][:NTOP]
print(f"top-{NTOP} признаков по gain отобраны", flush=True)
del m, Xtr_full

def prep(a, mode):
    Z = np.clip((X[a][:, top] - mu[top]) / sd[top], -8, 8).astype(np.float32)
    if mode == "rank":
        Z = np.argsort(np.argsort(Z, 0), 0).astype(np.float32) / NU
    return Z

y_ev = target(SEL_EV)
ly_ev = np.log1p(y_ev)
ref = OUT / "h8_gauss.npy"
rv = np.load(ref) if ref.exists() else None

print(f"\n{'рука':>10} {'k':>5} {'RMSLE':>9} {'corr с дневной':>15} {'сек':>7}")
RES = {}
for tag, mode, ks in (("knn", "std", (50, 200, 500)), ("knn_rank", "rank", (200,)),
                      ("knn_pca", "pca", (200,))):
    Ztr = np.concatenate([prep(a, "rank" if mode == "rank" else "std") for a in SEL_TR], 0)
    Zev = prep(SEL_EV, "rank" if mode == "rank" else "std")
    ytr_all = np.concatenate([np.log1p(target(a)) for a in SEL_TR])
    if mode == "pca":
        pca = PCA(n_components=24, random_state=0).fit(Ztr[::5])
        Ztr, Zev = pca.transform(Ztr).astype(np.float32), pca.transform(Zev).astype(np.float32)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=max(ks), algorithm="auto", n_jobs=11).fit(Ztr)
    dist, idx = nn.kneighbors(Zev)
    for k in ks:
        p = ytr_all[idx[:, :k]].mean(1)
        name = f"{tag}_k{k}"
        RES[name] = p
        c = np.corrcoef(p, rv)[0, 1] if rv is not None else float("nan")
        print(f"{tag:>10} {k:5d} {rmsle(y_ev, np.expm1(np.clip(p,0,13))):9.5f} {c:15.5f} "
              f"{time.time()-t0:7.0f}", flush=True)
        np.save(OUT / f"i2_{name}.npy", p)
    del Ztr, Zev, nn, dist, idx
print("\nвзаимные корреляции:")
ks_ = list(RES)
print("           " + " ".join(f"{k[:10]:>11}" for k in ks_))
for a_ in ks_:
    print(f"{a_:>10} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:11.4f}" for b in ks_))
print("\ndone")
