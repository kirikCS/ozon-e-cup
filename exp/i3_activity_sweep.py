"""Полный перебор направлений из активности + сборка одного оптимального направления.

i1 нашёл расслоение: покупочные переменные в остатке дают ноль (компоненты их выбрали до конца),
непокупочная активность держит t = -8,1 по visits_30. Здесь перебираются все направления,
которые можно построить из панелей, и решается, сколько независимой информации в них суммарно.

Экономия слотов — главное соображение. Зондировать четыре направления по отдельности значит
потратить четыре слота на то, что почти наверняка одна и та же ось: visits_30, engaged_30 и
sday_30 все меряют "клиент заходил". Вместо этого:

  1. считаем все направления, ортогонализуем каждое к {1, q};
  2. совместная ридж-регрессия остатка на все сразу — она сама разберётся, что дублирует;
  3. получившееся ОДНО направление и есть кандидат на зонд.

Локальные веса при этом не обязаны быть точными: от них нужно только направление, а его истинный
коэффициент измерит лидерборд. Любое направление с ненулевой проекцией на настоящее покажет
сигнал, поэтому конструкция устойчива к неточности локальной подгонки.

ПРОВЕРКА ПЕРЕНОСА обязательна. Направление подбирается на SELECT и применяется ЗАМОРОЖЕННЫМ на
HON. Локальный якорь в этом проекте уже трижды оказывался анти-предиктором, так что направление,
не переносящееся между двумя протоколами, зондировать нельзя.
"""
import sys
import numpy as np

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT

WIN = (7, 14, 30, 60, 90, 180)
CHAN = ["visit", "engaged", "srch", "cart", "sday", "catday", "ord", "gmv", "scart", "sord"]
PURCHASE = {"ord", "gmv", "sord"}

P = {c: panel(c).astype(np.float32) for c in CHAN}

def wsum(c, a, w):
    return P[c][:, max(0, a - w + 1):a + 1].sum(1)

def build_dirs(a):
    """все направления из активности; имена помечены, покупочное оно или нет"""
    D = {}
    for c in CHAN:
        for w in WIN:
            D[f"{c}_{w}"] = np.log1p(wsum(c, a, w))
    # интенсивность на активный день: сколько делает, когда пришёл
    for c in ("srch", "cart", "gmv"):
        for w in (30, 90):
            days = np.maximum(wsum("visit", a, w), 1)
            D[f"{c}_per_day_{w}"] = np.log1p(wsum(c, a, w) / days)
    # конверсии
    for num, den, nm in (("cart", "srch", "cart_per_srch"), ("ord", "cart", "ord_per_cart"),
                         ("sday", "visit", "srchday_share")):
        for w in (30, 90):
            D[f"{nm}_{w}"] = wsum(num, a, w) / np.maximum(wsum(den, a, w), 1)
    # свежесть: сколько дней назад было последнее событие
    for c in ("visit", "cart", "ord"):
        M = P[c][:, :a + 1] > 0
        last = a - (M.shape[1] - 1 - np.argmax(M[:, ::-1], 1))
        D[f"recency_{c}"] = np.log1p(np.where(M.any(1), last, a + 1))
    # регулярность: доля активных дней и тренд
    for w in (30, 90, 180):
        D[f"active_share_{w}"] = (P["visit"][:, max(0, a - w + 1):a + 1] > 0).mean(1)
    for c in ("visit", "srch", "cart"):
        D[f"trend_{c}"] = np.log1p(wsum(c, a, 30)) - np.log1p(wsum(c, a, 90) / 3.0)
    # доля каталога в активности
    D["cat_share_90"] = wsum("catday", a, 90) / np.maximum(wsum("visit", a, 90), 1)
    return D

def orth(d, basis):
    d = d - d.mean()
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

def residual(npy, a):
    p = np.load(OUT / npy)
    ly = np.log1p(target(a))
    p = p - p.mean() + ly.mean()
    return ly - np.clip(p, 0, 13), p - p.mean(), rmsle(target(a), np.expm1(np.clip(p, 0, 13)))

SRC = {"SELECT": ("h8_gauss.npy", 348), "HON": ("h5_base_hon.npy", 288)}
avail = {k: v for k, v in SRC.items() if (OUT / v[0]).exists()}
print(f"доступные протоколы: {list(avail)}")

R = {}
for proto, (npy, a) in avail.items():
    res, qc, sc = residual(npy, a)
    D = build_dirs(a)
    R[proto] = (res, qc, D, a, sc)
    print(f"  {proto} (якорь {a}): RMSLE {sc:.5f}, направлений {len(D)}")

proto0 = "SELECT"
res0, qc0, D0, a0, _ = R[proto0]
one = np.ones(NU)
ORTH0 = {}
for nm, v in D0.items():
    d = orth(v, [one, qc0])
    if d is not None:
        ORTH0[nm] = d
print(f"\n=== одиночные направления на {proto0}, топ-16 по |t| ===")
print(f"{'направление':>18} {'corr':>10} {'t':>8} {'тип':>12}")
rows = []
for nm, d in ORTH0.items():
    c = float(np.corrcoef(res0, d)[0, 1])
    t = c * np.sqrt(NU - 2) / np.sqrt(max(1 - c * c, 1e-12))
    kind = "покупочное" if any(p in nm for p in PURCHASE) else "активность"
    rows.append((abs(t), nm, c, t, kind))
for _, nm, c, t, kind in sorted(rows, reverse=True)[:16]:
    print(f"{nm:>18} {c:+10.5f} {t:8.1f} {kind:>12}")
np_ = [r for r in rows if r[4] == "покупочное"]
ac = [r for r in rows if r[4] == "активность"]
print(f"\nсредний |t|: покупочные {np.mean([r[0] for r in np_]):.1f} "
      f"(n={len(np_)}), активность {np.mean([r[0] for r in ac]):.1f} (n={len(ac)})")

print(f"\n=== совместное направление (ридж по всем сразу) ===")
names = list(ORTH0)
Aмат = np.stack([ORTH0[n] for n in names], 1)
lam = 1e-2 * len(names)
w = np.linalg.solve(Aмат.T @ Aмат + lam * np.eye(len(names)), Aмат.T @ res0)
comb0 = Aмат @ w
comb0 = comb0 / comb0.std()
c0 = float(np.corrcoef(res0, comb0)[0, 1])
print(f"  corr(остаток, совместное) на {proto0}: {c0:+.5f}, "
      f"t = {c0*np.sqrt(NU-2)/np.sqrt(max(1-c0*c0,1e-12)):.1f}")
print(f"  вклад топ-8 весов: " + ", ".join(
    f"{n} {ww:+.3f}" for ww, n in sorted(zip(w, names), key=lambda x: -abs(x[0]))[:8]))

if "HON" in R:
    res1, qc1, D1, a1, _ = R["HON"]
    ORTH1 = {}
    for nm, v in D1.items():
        d = orth(v, [np.ones(NU), qc1])
        if d is not None:
            ORTH1[nm] = d
    common = [n for n in names if n in ORTH1]
    wc = np.array([w[names.index(n)] for n in common])
    comb1 = np.stack([ORTH1[n] for n in common], 1) @ wc
    comb1 = comb1 / max(comb1.std(), 1e-9)
    c1 = float(np.corrcoef(res1, comb1)[0, 1])
    print(f"\n  ПЕРЕНОС: направление с SELECT, применённое замороженным на HON:")
    print(f"    corr(остаток, направление) = {c1:+.5f}, "
          f"t = {c1*np.sqrt(NU-2)/np.sqrt(max(1-c1*c1,1e-12)):.1f}")
    print(f"    на SELECT было {c0:+.5f} — {'переносится' if c1*c0 > 0 and abs(c1) > 0.3*abs(c0) else 'НЕ ПЕРЕНОСИТСЯ'}")
    best_single = sorted(rows, reverse=True)[0][1]
    if best_single in ORTH1:
        cs = float(np.corrcoef(res1, ORTH1[best_single])[0, 1])
        cs0 = float(np.corrcoef(res0, ORTH0[best_single])[0, 1])
        print(f"    для сравнения одиночное {best_single}: SELECT {cs0:+.5f}, HON {cs:+.5f}")
np.save(OUT / "i3_comb_dir_weights.npy", np.array([w[names.index(n)] for n in names]))
with open(OUT / "i3_dir_names.txt", "w") as f:
    f.write("\n".join(names))
print("\nвеса совместного направления сохранены")
print("done")
