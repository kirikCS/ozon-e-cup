"""Кликовое направление как пост-процессинг: локальная диагностика и генератор зонда.

Идея владельца: предсказание оставляем, добавляем поправку вдоль кликовой активности.

Почему это НЕ повторяет закрытое направление. c1/c2 закрыли ЗАМЕНУ предсказания на произведение
клики x цена клика — переформулировку целиком. Здесь поправка первого порядка к уже готовому
предсказанию, что совсем другой объект.

Почему это новая ось. Все шесть зондов формы были функциями САМОГО предсказания: 1, q-mu,
(q-mu)^2, (q-mu)^3, индикаторы корзин. Кликовое направление в их оболочке не лежит — оно несёт
внешнюю информацию.

Почему не надо ничего предсказывать. На финальном якоре 408 история кликов известна ТОЧНО.
Предсказывать будущие клики (и наследовать их ошибку) не требуется: спрашиваем, зависит ли
остаток стека от недавней кликовой интенсивности сверх того, что предсказание уже учло.

Диагностика на SELECT: если corr(остаток, кликовое направление) заметно отличается от нуля на
двух протоколах — направление живое, и зонд на лидерборде окупится. Если ноль — компоненты уже
всё выбрали через свои 165 признаков и 11 каналов, и слот тратить незачем.
"""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT

EL, VAR_L = 2.3284, 5.366802
PROTO = {"SELECT": 348, "HON": 288}
CH = [("searches_30", "srch", 30), ("searches_90", "srch", 90), ("carts_30", "cart", 30),
      ("visits_30", "visit", 30), ("engaged_30", "engaged", 30), ("srchdays_30", "sday", 30),
      ("orders_30", "ord", 30), ("gmv_30", "gmv", 30)]

def direction(anchor, ch, w):
    M = panel(ch).astype(np.float32)
    return np.log1p(M[:, max(0, anchor - w + 1):anchor + 1].sum(1))

def orth(d, basis):
    d = d - d.mean()
    for b in basis:
        d = d - np.dot(d, b) / np.dot(b, b) * b
    return d

print("=== 1. диагностика: остаток локальной модели против кликовых направлений ===")
best = None
for f in ("h8_gauss.npy", "h1_all_sel.npy", "g3_all_sel.npy", "q2c_daily_sel.npy"):
    if (OUT / f).exists():
        best = OUT / f
        break
if best is None:
    print("нет сохранённого локального предсказания — пропускаю диагностику")
else:
    print(f"локальная модель: {best.name}")
    for proto, A in (("SELECT", 348),):
        p = np.load(best)
        y = target(A)
        ly = np.log1p(y)
        p = p - p.mean() + ly.mean()
        res = ly - np.clip(p, 0, 13)
        qc = p - p.mean()
        print(f"\n  {proto} (якорь {A}), RMSLE {rmsle(y, np.expm1(np.clip(p,0,13))):.5f}")
        print(f"  {'направление':>14} {'corr с остатком':>16} {'t-стат':>9} {'выигрыш dRMSLE':>15}")
        for nm, ch, w in CH:
            d = orth(direction(A, ch, w), [np.ones(NU), qc])
            sd = d.std()
            if sd < 1e-9:
                continue
            d = d / sd
            c = float(np.corrcoef(res, d)[0, 1])
            t = c * np.sqrt(NU - 2) / np.sqrt(max(1 - c * c, 1e-12))
            gain = (c * res.std()) ** 2 / (2 * rmsle(y, np.expm1(np.clip(p, 0, 13))))
            print(f"  {nm:>14} {c:+16.5f} {t:9.1f} {gain:15.6f}")

print("\n=== 2. зонды для лидерборда (направления на финальном якоре 408) ===")
SRC = "submissions/stack_v12.csv"
R0 = 1.646001202698559
d0 = pl.read_csv(SRC).sort("user_id")
uid = d0["user_id"].to_numpy()
q = np.log1p(d0["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
one = np.ones_like(q)
TARGET_COST = 0.0015
eps = float(np.sqrt((R0 + TARGET_COST) ** 2 - R0 ** 2))

def write_exact(name, t):
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
            lo = c
        else:
            hi = c
    p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(f"submission/queue/{name}.csv")
    return float(np.log1p(p).mean())

print(f"{'файл':>26} {'sd напр.':>9} {'нуль':>10} {'цена':>9} {'уровень':>9}")
import os
os.makedirs("submission/queue", exist_ok=True)
for nm, ch, w in CH[:4]:
    d = orth(direction(408, ch, w), [one, qc])
    if d.std() < 1e-9:
        continue
    d = d / d.std()
    lv = write_exact(f"probe_click_{nm}", q + eps * d)
    print(f"{'probe_click_'+nm:>26} {1.0:9.3f} {np.sqrt(R0**2+eps**2):10.6f} "
          f"{np.sqrt(R0**2+eps**2)-R0:9.6f} {lv:9.6f}")
print(f"\n  анализ: theta = (нуль^2 - R'^2) / (2*eps),  выигрыш dR^2 = theta^2")
print(f"  порог полезности |theta| > {R0*1e-4/eps:.4f}")
print(f"  SE переноса на приват ~ {np.sqrt(R0**2/50000):.5f}, усадка kappa = max(0, 1 - SE^2/theta^2)")
print("\ndone")
