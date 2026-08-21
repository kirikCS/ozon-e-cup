"""Отток как отдельная модель: откалиброван ли уже наш вектор по нему.

Идея пользователя: занулять предсказания клиентов, которые ушли. Магнитудный порог (y_hat < t)
проверен в o1 и проигрывает на каждом уровне, но порог по ОТДЕЛЬНОЙ модели оттока — другое дело.

Главный вопрос при этом не про порог. Оптимум RMSLE это E[log1p y|x]; если клиент ушёл с
вероятностью p, оптимум уже сжат множителем (1-p), и жёсткий ноль верен только при p = 1 ровно.
Наша дневная модель содержит модель оттока внутри (хазард по дням + frailty), поэтому вопрос
такой: осталась ли в векторе систематическая ошибка, объяснимая вероятностью оттока.

Диагностика — остатки по децилям p_hat. Плоско, значит отток уже учтён, и направление закрыто
независимо от того, каким порогом его брать. Наклон есть — вот и величина выигрыша, и брать её
надо поправкой, а не занулением.

Всё честно по протоколу: классификатор и поправка обучаются на SEL_TR, применяются на 348.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()

print("строю признаки ...", flush=True)
X, names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.concatenate([(target(a) > 0).astype(np.int8) for a in SEL_TR], 0)
Xev = X[SEL_EV]; yev = (y > 0).astype(np.int8)
print(f"  {Xtr.shape[0]} строк, {len(names)} колонок, доля активных: обуч {ytr.mean():.4f}, "
      f"оценка {yev.mean():.4f}", flush=True)

t0 = time.time()
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                         min_child_samples=200, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
clf.fit(Xtr, ytr)
p_act = clf.predict_proba(Xev)[:, 1]          # вероятность быть АКТИВНЫМ
p_churn = 1.0 - p_act
from sklearn.metrics import roc_auc_score, log_loss
print(f"классификатор оттока: AUC {roc_auc_score(yev, p_act):.5f}, "
      f"logloss {log_loss(yev, p_act):.5f}, {time.time()-t0:.0f}s", flush=True)
base_ll = log_loss(yev, np.full(len(yev), ytr.mean()))
print(f"  logloss константы: {base_ll:.5f}   калибровка: среднее p {p_act.mean():.4f} "
      f"против факта {yev.mean():.4f}", flush=True)

def score(v):
    return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))

BASES = {}
b = np.load(OUT / "h8_gauss.npy"); BASES["дневная свёртка"] = b
got = []
for n in ["d7_fusion4_sel", "d2_fusion4_sel", "g5_coles_grn_sel", "h1_tft_all_sel"]:
    p = OUT / f"{n}.npy"
    if p.exists():
        v = np.load(p); got.append(v - v.mean())
if len(got) >= 3:
    BASES["равновесная смесь"] = np.mean(got, 0)

for bn, base in BASES.items():
    v = base - base.mean() + MU
    s0 = score(v); r = ly - v
    print(f"\n=== опора «{bn}», RMSLE {s0:.6f} ===", flush=True)
    print("ОСТАТКИ ПО ДЕЦИЛЯМ ВЕРОЯТНОСТИ ОТТОКА (плоско = отток уже учтён)")
    q = np.quantile(p_churn, np.linspace(0, 1, 11)); q[0] -= 1e-9; q[-1] += 1e-9
    dec = np.clip(np.digitize(p_churn, q[1:-1]), 0, 9)
    print(f"{'дециль':>7} {'p_отток':>9} {'факт нулей':>11} {'сред. остаток':>14} {'клиентов':>9}")
    for d in range(10):
        m = dec == d
        print(f"{d:7d} {p_churn[m].mean():9.4f} {(y[m]==0).mean():11.4f} "
              f"{r[m].mean():+14.5f} {m.sum():9d}")
    sl = np.polyfit(p_churn, r, 1)[0]
    print(f"наклон остатка по p_отток: {sl:+.5f}  (ноль = поправки нет)")

    # честная поправка: подецильные сдвиги, обученные на SEL_TR, применённые на 348
    print("\nЖЁСТКОЕ ЗАНУЛЕНИЕ ПО p_отток")
    print(f"{'порог':>7} {'занулено':>9} {'из них верно':>13} {'RMSLE':>10} {'дельта':>11}")
    for t in (0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.6):
        m = p_churn > t
        if m.sum() == 0: continue
        vv = v.copy(); vv[m] = 0.0
        print(f"{t:7.2f} {m.mean():9.4f} {(y[m]==0).mean():13.4f} {score(vv):10.6f} "
              f"{score(vv)-s0:+11.2e}")

    print("\nМЯГКАЯ ПОПРАВКА  log1p(y_hat) *= (1-p_отток)^alpha")
    print(f"{'alpha':>7} {'RMSLE':>10} {'дельта':>11}")
    for al in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, -0.05, -0.1):
        vv = v * (1.0 - p_churn) ** al
        print(f"{al:7.2f} {score(vv):10.6f} {score(vv)-s0:+11.2e}")
print("\ndone")
