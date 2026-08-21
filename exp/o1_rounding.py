"""Рекурсивное округление знаков после запятой как пост-процессинг.

Цель НЕ квантована: 140 721 уникальное ненулевое значение, ни одного круглого, минимум 0,031.
Сетки, к которой можно притянуть, не существует, поэтому округление — независимая добавка к
предсказанию, и её вред считается заранее. Равномерное квантование с шагом h даёт дисперсию
h^2/12, а RMSLE считается в пространстве log1p, поэтому:

  MSE_new = MSE_old + 2*E[d*r] + E[d^2],  r = log1p(y_hat) - log1p(y)

При d, не связанном с остатком, средний член зануляется перекалибровкой уровня, и остаётся строго
положительное E[d^2]. Предсказание печатается рядом с замером — это и есть проверка.

Рекурсия: округлить -> подогнать уровень -> округлить, до неподвижной точки. Сдвиг уровня уводит
значения с сетки, поэтому итерация нетривиальна и может не сходиться.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from common import target, rmsle, OUT

y = target(348)
ly = np.log1p(y)
MU = ly.mean()

def score(v):
    return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))

names = ["h8_gauss", "d7_fusion4_sel", "d2_fusion4_sel", "g5_coles_grn_sel", "h1_tft_all_sel"]
BASES = {}
b = np.load(OUT / "h8_gauss.npy"); BASES["дневная свёртка"] = b
got = []
for n in names:
    p = OUT / f"{n}.npy"
    if p.exists():
        v = np.load(p); got.append(v - v.mean())
if len(got) >= 3:
    BASES["равновесная смесь"] = np.mean(got, 0)
for k, v in BASES.items():
    print(f"опора «{k}»: RMSLE {score(v):.6f}", flush=True)

def refit(v):
    return v - v.mean() + MU

print(f"\n=== A. округление в СЫРОМ пространстве ===")
print(f"{'знаков':>7} {'шаг h':>10} {'RMSLE':>10} {'дельта':>11} {'предсказано':>12}")
for base_name, base in BASES.items():
    print(f"-- {base_name}")
    v = refit(base); s0 = score(v); mse0 = s0 ** 2
    yh = np.expm1(np.clip(v, 0, 13))
    for k in range(6, -4, -1):
        h = 10.0 ** (-k)
        q = np.log1p(np.maximum(np.round(yh, k), 0))
        s = score(q)
        # предсказание: d ~ U(-h/2,h/2) в сыром, в log1p делится на (1+y_hat)
        pv = (h ** 2 / 12) * np.mean(1.0 / (1.0 + yh) ** 2)
        print(f"{k:7d} {h:10.4g} {s:10.6f} {s-s0:+11.2e} {np.sqrt(mse0+pv)-s0:+12.2e}")

print(f"\n=== B. округление в пространстве log1p ===")
print(f"{'шаг h':>10} {'RMSLE':>10} {'дельта':>11} {'предсказано':>12}")
for base_name, base in BASES.items():
    print(f"-- {base_name}")
    v = refit(base); s0 = score(v); mse0 = s0 ** 2
    for h in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001):
        q = np.round(v / h) * h
        s = score(q)
        print(f"{h:10.4g} {s:10.6f} {s-s0:+11.2e} {np.sqrt(mse0+h*h/12)-s0:+12.2e}")

print(f"\n=== C. РЕКУРСИЯ: округлить -> подогнать уровень -> округлить ===")
for base_name, base in BASES.items():
    print(f"-- {base_name}")
    for h in (0.5, 0.1, 0.05):
        v = refit(base); traj = []
        seen = {}
        for it in range(30):
            v = np.round(v / h) * h
            v = v - v.mean() + MU
            traj.append(score(v))
            key = v.tobytes()[:4000]
            if key in seen:
                print(f"  h={h:<6g} цикл на шаге {it}, длина {it-seen[key]}, "
                      f"скоры {['%.6f'%t for t in traj[:6]]} ... {traj[-1]:.6f}")
                break
            seen[key] = it
        else:
            print(f"  h={h:<6g} без цикла за 30 шагов, скоры {['%.6f'%t for t in traj[:4]]} "
                  f"... {traj[-1]:.6f}")
        print(f"           лучший по траектории {min(traj):.6f} на шаге {int(np.argmin(traj))}, "
              f"опора {score(refit(base)):.6f}")

print(f"\n=== D. порог в ноль (единственный атом, что есть: P(y=0)=0,4369) ===")
for base_name, base in BASES.items():
    v = refit(base); s0 = score(v)
    yh = np.expm1(np.clip(v, 0, 13))
    print(f"-- {base_name}, опора {s0:.6f}")
    best = (s0, 0.0)
    for t in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        z = np.where(yh < t, 0.0, yh)
        s = score(np.log1p(z))
        if s < best[0]: best = (s, t)
        print(f"  порог {t:6.2f}: {s:.6f}  ({s-s0:+.2e}), обнулено {(yh<t).mean():.4f}")
    print(f"  лучший порог {best[1]} -> {best[0]:.6f}")
print("\ndone")
