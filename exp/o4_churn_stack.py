"""Не дублирует ли поправка на отток то, что бустинг на фичесторе уже знает.

В o3 поправка выжила честное разделение по клиентам (-8,6e-5 подецильными сдвигами), но опорой
была одиночная дневная свёртка. Вероятность оттока считается по фичестору, а в стеке на том же
фичесторе сидят бустинги — если они это уже знают, на стеке поправка даст ноль.

Здесь база собирается БЕЗ подгонки на оценочном якоре: равновесное среднее нейронных векторов и
обученного тут же бустинга. Веса не подбираются, значит утечки нет. Дальше те же три поправки,
и снова подгонка на половине клиентов с оценкой на другой, обе стороны.

Второй классификатор оттока обучен на непересекающемся наборе якорей — если поправка держится и
на нём, она не артефакт конкретного разбиения.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
ALT_TR = [318, 262, 206]
y = target(SEL_EV); ly = np.log1p(y)

print("фичестор ...", flush=True)
ANCH = sorted(set(SEL_TR + ALT_TR + [SEL_EV]))
X, names = build_features(ANCH, verbose=False)

def fit_lgb(TR, obj):
    Xtr = np.concatenate([X[a] for a in TR], 0)
    if obj == "reg":
        ytr = np.concatenate([np.log1p(target(a)) for a in TR], 0)
        m = lgb.LGBMRegressor(n_estimators=900, learning_rate=0.04, num_leaves=95,
                              min_child_samples=150, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    else:
        ytr = np.concatenate([(target(a) > 0).astype(np.int8) for a in TR], 0)
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    return m

t0 = time.time()
reg = fit_lgb(SEL_TR, "reg")
v_lgb = reg.predict(X[SEL_EV]).astype(np.float64)
def sc0(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
print(f"бустинг на фичесторе: {sc0(v_lgb):.5f}  ({time.time()-t0:.0f}s)", flush=True)

PCH = {}
for tag, TR in (("основной", SEL_TR), ("на других якорях", ALT_TR)):
    PCH[tag] = 1.0 - fit_lgb(TR, "clf").predict_proba(X[SEL_EV])[:, 1]
    print(f"классификатор оттока [{tag}]: AUC "
          f"{roc_auc_score((y>0).astype(int), 1-PCH[tag]):.5f}", flush=True)
print(f"corr между двумя p_отток: {np.corrcoef(PCH['основной'], PCH['на других якорях'])[0,1]:.5f}",
      flush=True)

NEUR = ["h8_gauss", "d7_fusion4_sel", "d6_fusion_sel", "c5_sel", "d2_fusion4_sel"]
neur = [np.load(OUT / f"{n}.npy") for n in NEUR if (OUT / f"{n}.npy").exists()]
neur = [v - v.mean() for v in neur]
BASES = {
    "только нейронная":        np.mean(neur, 0),
    "нейронная + бустинг":     0.5*np.mean(neur, 0) + 0.5*(v_lgb - v_lgb.mean()),
}
for k, v in BASES.items():
    print(f"опора «{k}»: {sc0(v):.5f}", flush=True)

# сколько оттока бустинг уже знает: связь p_отток с ЕГО остатком
rl = ly - (v_lgb - v_lgb.mean() + ly.mean())
print(f"\ncorr(p_отток, остаток бустинга) = {np.corrcoef(PCH['основной'], rl)[0,1]:+.5f}",
      flush=True)

def affine(v, m):
    A = np.stack([v[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, ly[m], rcond=None)[0]
    return a*v + b
def sc(v, m): return rmsle(y[m], np.expm1(np.clip(v[m], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
for ptag, pch in PCH.items():
    print(f"\n{'='*70}\nклассификатор [{ptag}]")
    print(f"{'опора':>22} {'поправка':>22} {'A→B':>11} {'B→A':>11} {'среднее':>11}")
    for bname, v0 in BASES.items():
        for tname in ("жёсткий ноль", "мягкая (1-p)^alpha", "подецильные сдвиги"):
            d = []
            for fm, em in ((half, ~half), (~half, half)):
                vb = affine(v0, fm); s0 = sc(vb, em)
                if tname == "жёсткий ноль":
                    bt = min((0.99,0.98,0.97,0.95,0.92,0.90),
                             key=lambda t: sc(np.where(pch>t, 0.0, vb), fm))
                    w = np.where(pch > bt, 0.0, vb)
                elif tname.startswith("мягкая"):
                    ba = min(np.arange(-0.15, 0.31, 0.01),
                             key=lambda a: sc(vb*(1-pch)**a, fm))
                    w = vb*(1-pch)**ba
                else:
                    q = np.quantile(pch, np.linspace(0,1,11)); q[0]-=1e-9; q[-1]+=1e-9
                    dd = np.clip(np.digitize(pch, q[1:-1]), 0, 9)
                    w = vb.copy()
                    for k in range(10):
                        m = dd == k
                        w[m] += (ly[m & fm] - vb[m & fm]).mean()
                d.append(sc(w, em) - s0)
            print(f"{bname:>22} {tname:>22} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e}")
print("\nотрицательное = помогло на невиданных клиентах\ndone")
