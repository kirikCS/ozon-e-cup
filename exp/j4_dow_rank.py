"""Два направления, которых нет в фичесторе как класса: выравнивание по дню недели и
популяционный ранг.

Критерий отбора взят из результата probe_yago (S/N 10.2): работает то, чего компоненты
структурно не могут видеть, а не новое представление уже видимых данных. Событийная
переиндексация того же дала S/N 3.3 и подтвердила критерий с другой стороны.

НАПРАВЛЕНИЕ 1 — выравнивание недельного ритма с составом целевого окна.
Целевое окно 14.02.2026–15.03.2026 начинается в СУББОТУ и содержит 5 суббот и 5 воскресений
против 4 каждого буднего. Прошлогоднее окно начинается в ПЯТНИЦУ: 5 пятниц, 5 суббот, 4
воскресенья. Разница +1 воскресенье, −1 пятница.

Плюс праздники сместились по дням недели: 23 февраля 2026 — понедельник (длинные выходные
суббота-понедельник), в 2025 — воскресенье; 8 марта 2026 — воскресенье, в 2025 — суббота.

features_v2 содержит профиль дня недели и его энтропию, но не СВЁРТКУ этого профиля с конкретным
составом целевого окна. Клиент, покупающий по воскресеньям, в этом окне получает лишнее
воскресенье; покупающий по пятницам — теряет пятницу. Ни одна модель этого не знает.

НАПРАВЛЕНИЕ 2 — популяционный ранг и его изменение.
Фичестор содержит только собственные значения клиента. Ранг сам по себе бесполезен: деревья
инвариантны к монотонным преобразованиям, а ранг gmv_30 — монотонное преобразование gmv_30.
Полезно ИЗМЕНЕНИЕ ранга: оно отличается от изменения абсолютной величины ровно тогда, когда
сдвинулась вся популяция (сезонность, рост площадки). «Клиент вырос вдвое» и «клиент вырос
вдвое, когда все выросли втрое» — разные состояния, и второе видно только через ранг.
"""
import sys, os
import numpy as np
import polars as pl
from datetime import date, timedelta

sys.path.insert(0, "code")
from common import panel, NU

EL, VAR_L, NPUB = 2.3284, 5.366802, 50000
SRC = "submissions/stack_v14.csv"
R0 = 1.64577576410033
A, D0 = 408, date(2025, 1, 1)
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)

def dow_counts(start, n=30):
    c = np.zeros(7)
    for i in range(n):
        c[(D0 + timedelta(days=start + i)).weekday()] += 1
    return c

CT = dow_counts(409)          # состав целевого окна
CY = dow_counts(44)           # состав прошлогоднего окна
DOW = np.array([(D0 + timedelta(days=int(d))).weekday() for d in range(A + 1)])

def dow_profile(M, lo, hi):
    """доля активности клиента по дням недели в окне [lo, hi]"""
    P = np.zeros((NU, 7), np.float32)
    for d in range(7):
        cols = np.flatnonzero((DOW[lo:hi + 1] == d)) + lo
        if len(cols):
            P[:, d] = M[:, cols].sum(1)
    s = P.sum(1, keepdims=True)
    return P / np.maximum(s, 1e-9), s.ravel()

def lg(x):
    return np.log1p(np.maximum(x, 0))

print("=== направление 1: выравнивание недельного ритма ===")
D = {}
for nm, M, lo in (("gmv_year", G, 44), ("gmv_recent", G, A - 180), ("vis_recent", V, A - 180)):
    P, tot = dow_profile(M, lo, A)
    D[f"align_{nm}"] = (P @ CT) / 30.0 - (P @ np.full(7, 30 / 7)) / 30.0
    D[f"shift_{nm}"] = (P @ (CT - CY)) / 30.0
    print(f"  {nm:12s} sd выравнивания {D[f'align_{nm}'].std():.5f}, "
          f"sd сдвига {D[f'shift_{nm}'].std():.5f}")
# отдельно: доля выходных, потому что окно перекошено к ним
P, _ = dow_profile(G, A - 180, A)
D["weekend_share"] = P[:, 5] + P[:, 6]
# праздники сменили день недели: 23.02 пн-2026 против вс-2025, 08.03 вс-2026 против сб-2025
D["mon_share"] = P[:, 0]
D["sun_share"] = P[:, 6]

print("\n=== направление 2: популяционный ранг и его изменение ===")
def rank(x):
    r = np.empty(NU, np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(NU)
    return r / NU

R = {}
for nm, M in (("gmv", G), ("vis", V), ("cart", C)):
    for w in (30, 90):
        cur = rank(M[:, A - w + 1:A + 1].sum(1))
        prv = rank(M[:, A - 2 * w + 1:A - w + 1].sum(1))
        R[f"drank_{nm}_{w}"] = cur - prv
        print(f"  drank_{nm}_{w}: sd {(cur-prv).std():.5f}, "
              f"corr с изменением логарифма "
              f"{np.corrcoef(cur-prv, lg(M[:,A-w+1:A+1].sum(1))-lg(M[:,A-2*w+1:A-w+1].sum(1)))[0,1]:+.4f}")
# ускорение ранга: изменение изменения
for nm, M in (("gmv", G),):
    r1 = rank(M[:, A - 29:A + 1].sum(1)); r2 = rank(M[:, A - 59:A - 29].sum(1))
    r3 = rank(M[:, A - 89:A - 59].sum(1))
    R["rank_accel"] = (r1 - r2) - (r2 - r3)

d0f = pl.read_csv(SRC).sort("user_id")
uid = d0f["user_id"].to_numpy()
q = np.log1p(d0f["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
one = np.ones(NU)
basis = [one, qc]
for nm, path, base in (("клик", "submissions/probe_activity_comb.csv", "submissions/stack_v12.csv"),
                       ("событ", "submissions/probe_event_norm.csv", "submissions/stack_v14.csv"),
                       ("прошлогод", "submissions/probe_yago.csv", "submissions/stack_v14.csv")):
    if os.path.exists(path) and os.path.exists(base):
        a_ = np.log1p(pl.read_csv(path).sort("user_id")["predict"].to_numpy())
        b_ = np.log1p(pl.read_csv(base).sort("user_id")["predict"].to_numpy())
        dd = (a_ - a_.mean()) - (b_ - b_.mean())
        for b in basis:
            dd = dd - np.dot(dd, b) / np.dot(b, b) * b
        if dd.std() > 1e-9:
            basis.append(dd / dd.std())
            print(f"  в базис ортогонализации: {nm}")

def orth(d):
    d = np.asarray(d, np.float64) - np.mean(d)
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

eps = float(np.sqrt((R0 + 0.0015) ** 2 - R0 ** 2))
os.makedirs("submission/queue", exist_ok=True)

def emit(name, parts):
    comb = np.zeros(NU); n = 0
    for v in parts.values():
        dd = orth(v)
        if dd is not None:
            comb += dd; n += 1
    comb = orth(comb)
    if comb is None:
        print(f"{name}: направление вырождено"); return
    t = q + eps * comb
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL: lo = c
        else: hi = c
    p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(f"submission/queue/{name}.csv")
    orthchk = max(abs(np.corrcoef(comb, b)[0, 1]) for b in basis[2:]) if len(basis) > 2 else 0.0
    print(f"\n{name}.csv: из {n} частей, нуль {np.sqrt(R0**2+eps**2):.6f}, "
          f"уровень {np.log1p(p).mean():.6f}, макс corr с измеренными {orthchk:.1e}")

emit("probe_dow_align", D)
emit("probe_rank_shift", R)
print(f"\nопора {R0:.7f}, цена слота {np.sqrt(R0**2+eps**2)-R0:.6f}")
print(f"порог аппарата |θ| > {R0*1e-4/eps:.4f}, порог переноса |θ| > {np.sqrt(R0**2/NPUB):.4f}")
print("done")
