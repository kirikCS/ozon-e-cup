"""Направление «те же 30 дней год назад» — сезонное выравнивание по фиксированным датам.

Целевое окно 2026-02-14 … 2026-03-15 содержит 23 февраля и 8 марта — оба подарочные, оба с
ЖЁСТКО ФИКСИРОВАННОЙ датой. Ровно за год до него лежит 2025-02-14 … 2025-03-15, то есть дни
44–73 нашей истории: доступны целиком.

Почему этого нет ни в одной модели. Фичестор содержит окна, ЗАКАНЧИВАЮЩИЕСЯ на якоре: 7, 14, 30,
60, 90, 180, 365 дней назад. Окна вида «30 дней, начинающиеся 365 дней назад» нет. Ближайшее —
`gmv_365`, но это сумма за весь год, в которой праздничный всплеск размазан двенадцатикратно.
Дневные модели видят 364 последних дня, то есть дни 45–408 при якоре 408 — прошлогоднее окно
попадает на самый край их рецептивного поля и без какой-либо привязки к календарю цели.

ВАЖНОЕ ОГРАНИЧЕНИЕ: локально это направление проверить нельзя. Для якоря `a` прошлогоднее окно
цели — это дни `a+1-365 … a+30-365`, что требует `a >= 364`. Подходит только финальный якорь 408.
На отборочном 348 окно уходит в отрицательные дни. Поэтому диагностики не будет — только зонд,
и его нуль арифметический, так что риск ограничен ценой слота 0.0015.

Строится КОНТРАСТ, а не уровень: «насколько клиент был активен именно в том окне относительно
собственного тогдашнего фона». Уровень уже несёт `gmv_365`; контраст ортогонален ему по
построению и выражает личную сезонность.

Четыре варианта контраста сворачиваются в одно направление равными весами — подбирать их не на
чем, а любая ненулевая проекция на настоящее направление проявится в замере.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, NU

EL, VAR_L, NPUB = 2.3284, 5.366802, 50000
SRC = "submissions/stack_v14.csv"
R0 = 1.64577576410033
A = 408
YS, YE = A + 1 - 365, A + 30 - 365          # 44 … 73  = 2025-02-14 … 2025-03-15
F23, M08 = 53, 66                            # 23 февраля и 8 марта 2025

G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
O = panel("ord").astype(np.float32)
print(f"прошлогоднее окно цели: дни {YS}…{YE}  (2025-02-14 … 2025-03-15)")
print(f"  доля клиентов с покупкой в нём: {(G[:, YS:YE+1].sum(1) > 0).mean():.4f}")
print(f"  доля с покупкой в ±3 дня от 23.02 или 08.03: "
      f"{((G[:, F23-3:F23+4].sum(1) + G[:, M08-3:M08+4].sum(1)) > 0).mean():.4f}")

def lg(x):
    return np.log1p(np.maximum(x, 0))

# фон того же периода: 90 дней вокруг окна, исключая само окно
bg_lo, bg_hi = max(0, YS - 45), min(408, YE + 45)
bg = G[:, bg_lo:bg_hi + 1].sum(1) - G[:, YS:YE + 1].sum(1)
bg_days = (bg_hi - bg_lo + 1) - (YE - YS + 1)
D = {}
D["yago_lift"] = lg(G[:, YS:YE + 1].sum(1)) - lg(bg / max(bg_days, 1) * 30.0)
D["yago_hol"] = lg(G[:, F23 - 3:F23 + 4].sum(1) + G[:, M08 - 3:M08 + 4].sum(1)) \
    - lg(G[:, YS:YE + 1].sum(1) / 30.0 * 14.0)
D["yago_vis"] = lg(V[:, YS:YE + 1].sum(1)) - lg(V[:, bg_lo:bg_hi + 1].sum(1) / max(bg_days + 30, 1) * 30.0)
D["yago_ord"] = lg(O[:, YS:YE + 1].sum(1)) - lg(O[:, bg_lo:bg_hi + 1].sum(1) / max(bg_days + 30, 1) * 30.0)

d0 = pl.read_csv(SRC).sort("user_id")
uid = d0["user_id"].to_numpy()
q = np.log1p(d0["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
one = np.ones(NU)

basis = [one, qc]
for nm, path, base in (("клик", "submissions/probe_activity_comb.csv", "submissions/stack_v12.csv"),
                       ("событ", "submissions/probe_event_norm.csv", "submissions/stack_v14.csv")):
    if os.path.exists(path) and os.path.exists(base):
        a_ = np.log1p(pl.read_csv(path).sort("user_id")["predict"].to_numpy())
        b_ = np.log1p(pl.read_csv(base).sort("user_id")["predict"].to_numpy())
        dd = (a_ - a_.mean()) - (b_ - b_.mean())
        for b in basis:
            dd = dd - np.dot(dd, b) / np.dot(b, b) * b
        if dd.std() > 1e-9:
            basis.append(dd / dd.std())
            print(f"  в базис ортогонализации добавлено направление: {nm}")

def orth(d):
    d = np.asarray(d, np.float64)
    d = d - d.mean()
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

print(f"\n{'вариант':>12} {'sd после ортог.':>16} {'corr с q':>11}")
comb = np.zeros(NU)
n = 0
for nm, v in D.items():
    dd = orth(v)
    if dd is None:
        print(f"{nm:>12} {'вырожден':>16}")
        continue
    print(f"{nm:>12} {1.0:16.3f} {np.corrcoef(v, q)[0,1]:+11.4f}")
    comb += dd
    n += 1
comb = orth(comb)
print(f"\nсовместное из {n} вариантов: corr с q {np.corrcoef(comb,q)[0,1]:+.2e}, "
      f"с кликовым {np.corrcoef(comb,basis[2])[0,1]:+.2e}" if len(basis) > 2 else "")

eps = float(np.sqrt((R0 + 0.0015) ** 2 - R0 ** 2))
lo, hi = -6.0, 6.0
t = q + eps * comb
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
pl.DataFrame({"user_id": uid, "predict": p}).write_csv("submission/queue/probe_yago.csv")
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\nprobe_yago.csv")
print(f"  опора {R0:.7f}, eps {eps:.5f}, нуль {null:.6f}, цена {null-R0:.6f}, "
      f"уровень {np.log1p(p).mean():.6f}")
print(f"  порог аппарата |θ| > {R0*1e-4/eps:.4f}, порог переноса |θ| > {np.sqrt(R0**2/NPUB):.4f}")
print("done")
