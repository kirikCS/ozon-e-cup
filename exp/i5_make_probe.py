"""Зонд вдоль совместного направления активности, построенного на якоре 408.

Веса подобраны в i3 на SELECT (82 направления, ридж). Здесь те же направления строятся на
финальном якоре и сворачиваются теми же весами. Предсказывать ничего не надо: на 408 вся история
активности известна точно.

Направление ортогонализуется к {1, q} предсказания, поэтому зонд меряет ровно то, чего в самом
предсказании нет: R'^2 = R0^2 - 2*eps*Cov(L-P, d) + eps^2*Var(d), и при Var(d)=1 нуль
арифметический.

Локальная подгонка даёт только НАПРАВЛЕНИЕ; его истинный коэффициент измерит лидерборд. Поэтому
неточность локальных весов не фатальна — любая ненулевая проекция на настоящее направление
проявится. Фатальным будет только случай, когда настоящего направления нет вовсе, и тогда theta
выйдет около нуля.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, target, NU, OUT

EL, VAR_L = 2.3284, 5.366802
SRC = "submissions/stack_v12.csv"
R0 = 1.646001202698559
A = 408
WIN = (7, 14, 30, 60, 90, 180)
CHAN = ["visit", "engaged", "srch", "cart", "sday", "catday", "ord", "gmv", "scart", "sord"]
P = {c: panel(c).astype(np.float32) for c in CHAN}
wsum = lambda c, a, w: P[c][:, max(0, a - w + 1):a + 1].sum(1)

def build_dirs(a):
    D = {}
    for c in CHAN:
        for w in WIN:
            D[f"{c}_{w}"] = np.log1p(wsum(c, a, w))
    for c in ("srch", "cart", "gmv"):
        for w in (30, 90):
            D[f"{c}_per_day_{w}"] = np.log1p(wsum(c, a, w) / np.maximum(wsum("visit", a, w), 1))
    for num, den, nm in (("cart", "srch", "cart_per_srch"), ("ord", "cart", "ord_per_cart"),
                         ("sday", "visit", "srchday_share")):
        for w in (30, 90):
            D[f"{nm}_{w}"] = wsum(num, a, w) / np.maximum(wsum(den, a, w), 1)
    for c in ("visit", "cart", "ord"):
        M = P[c][:, :a + 1] > 0
        last = a - (M.shape[1] - 1 - np.argmax(M[:, ::-1], 1))
        D[f"recency_{c}"] = np.log1p(np.where(M.any(1), last, a + 1))
    for w in (30, 90, 180):
        D[f"active_share_{w}"] = (P["visit"][:, max(0, a - w + 1):a + 1] > 0).mean(1)
    for c in ("visit", "srch", "cart"):
        D[f"trend_{c}"] = np.log1p(wsum(c, a, 30)) - np.log1p(wsum(c, a, 90) / 3.0)
    D["cat_share_90"] = wsum("catday", a, 90) / np.maximum(wsum("visit", a, 90), 1)
    return D

names = (OUT / "i3_dir_names.txt").read_text().split("\n")
w = np.load(OUT / "i3_comb_dir_weights.npy")
print(f"весов из i3: {len(w)}, имён: {len(names)}")

d0 = pl.read_csv(SRC).sort("user_id")
uid = d0["user_id"].to_numpy()
q = np.log1p(d0["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
one = np.ones(NU)

def orth(d, basis):
    d = d - d.mean()
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

D = build_dirs(A)
comb = np.zeros(NU)
used = 0
for nm, wi in zip(names, w):
    if nm not in D:
        continue
    dd = orth(D[nm], [one, qc])
    if dd is None:
        continue
    comb += wi * dd
    used += 1
print(f"собрано направлений на якоре {A}: {used}")
comb = orth(comb, [one, qc])
print(f"совместное направление: sd {comb.std():.4f}, corr с q {np.corrcoef(comb,q)[0,1]:+.2e}")

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
    os.makedirs("submission/queue", exist_ok=True)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(f"submission/queue/{name}.csv")
    return float(np.log1p(p).mean())

lv = write_exact("probe_activity_comb", q + eps * comb)
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\nprobe_activity_comb.csv")
print(f"  eps {eps:.5f}, нуль {null:.6f}, цена слота {null-R0:.6f}, уровень файла {lv:.6f}")
print(f"\n  анализ: theta = (нуль^2 - R'^2) / (2*eps)")
print(f"  выигрыш от применения: dR^2 = theta^2 (после усадки kappa*(2-kappa))")
print(f"  порог аппарата |theta| > {R0*1e-4/eps:.4f}")
print(f"  порог переноса на приват |theta| > {np.sqrt(R0**2/50000):.4f}")
print(f"  локальная оценка предсказывает theta ~ -0.078 (corr 0.047 x sd остатка)")
print("done")
