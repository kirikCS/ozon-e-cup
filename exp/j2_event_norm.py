"""Событийная нормировка: направления, индексированные событиями, а не календарными днями.

Кликовый зонд (i3/i5) дал theta = -0.02547, t = -3.46, S/N 13.2 — единственный шаг проекта выше
порога достоверности. Но все 82 его направления были КАЛЕНДАРНЫМИ окнами: сумма за последние
30/90/180 дней. Событийное время — другая ось, и она в проекте самая ортогональная: модель на
событийных часах (d_active) имеет g = 0.982, ниже всех в дневном семействе.

Главная величина, которой нет в календарных окнах, — НОРМИРОВАННАЯ СВЕЖЕСТЬ. Клиент с типичным
разрывом 3 дня, не заходивший 10 дней, находится в другом состоянии, чем клиент с типичным
разрывом 30 дней в той же ситуации. Календарное окно их не различает; отношение «дней с
последнего события / собственный типичный разрыв» — различает.

Строятся четыре группы:
  gap    статистики межсобытийных разрывов: среднее, медиана, sd, максимум, коэффициент вариации
  norm   свежесть, нормированная на собственный типичный разрыв; и то же для покупок
  burst  кучность (Fano, burstiness) — события идут пачками или ровно
  ewin   агрегаты по последним K СОБЫТИЯМ (8/16/32/64), а не дням, и контрасты между ними

КРИТИЧНО: новое направление ортогонализуется не только к {1, q}, но и к УЖЕ ИЗМЕРЕННОМУ
кликовому направлению. Иначе зонд частично перемерит theta, который уже известен, и потратит
слот на подтверждение известного.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT

EL, VAR_L, NPUB = 2.3284, 5.366802, 50000
CH = ["visit", "gmv", "ord", "cart", "srch"]
P = {c: panel(c).astype(np.float32) for c in CH}
ACT = P["visit"] > 0
BUY = P["gmv"] > 0

def event_dirs(a, K=64):
    """величины в событийном времени: разрывы, нормированная свежесть, кучность, окна по событиям"""
    D = {}
    m = ACT[:, :a + 1]
    rev = m[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K - cc[ui, jj]
    DAYS = np.full((NU, K), -1, np.int32)
    DAYS[ui, slot] = day
    valid = DAYS >= 0
    nev = valid.sum(1)
    # разрывы между соседними событиями
    d0 = DAYS.astype(np.float64)
    prev = np.roll(d0, 1, 1); pv = np.roll(valid, 1, 1)
    prev[:, 0] = -1; pv[:, 0] = False
    gaps = np.where(valid & pv, d0 - prev, np.nan)
    with np.errstate(invalid="ignore"):
        gm = np.nanmean(gaps, 1); gmed = np.nanmedian(gaps, 1)
        gsd = np.nanstd(gaps, 1); gmax = np.nanmax(gaps, 1)
    gm = np.nan_to_num(gm, nan=a + 1.0); gmed = np.nan_to_num(gmed, nan=a + 1.0)
    gsd = np.nan_to_num(gsd); gmax = np.nan_to_num(gmax, nan=a + 1.0)
    D["gap_mean"] = np.log1p(gm)
    D["gap_median"] = np.log1p(gmed)
    D["gap_sd"] = np.log1p(gsd)
    D["gap_max"] = np.log1p(gmax)
    D["gap_cv"] = gsd / np.maximum(gm, 1e-6)
    D["n_events"] = np.log1p(nev.astype(np.float64))
    # нормированная свежесть: сколько собственных типичных разрывов прошло
    last = np.where(valid.any(1), DAYS.max(1), -1)
    rec = np.where(last >= 0, a - last, a + 1).astype(np.float64)
    D["recency_days"] = np.log1p(rec)
    D["recency_norm_mean"] = np.log1p(rec / np.maximum(gm, 0.5))
    D["recency_norm_med"] = np.log1p(rec / np.maximum(gmed, 0.5))
    D["overdue"] = np.clip(rec / np.maximum(gmed, 0.5) - 1.0, -2, 20)
    # то же для покупок
    mb = BUY[:, :a + 1]
    lastb = np.where(mb.any(1), a - (mb.shape[1] - 1 - np.argmax(mb[:, ::-1], 1)), a + 1)
    nb = mb.sum(1).astype(np.float64)
    gapb = np.where(nb > 1, (a - np.argmax(mb, 1)) / np.maximum(nb - 1, 1), a + 1.0)
    D["buy_recency_norm"] = np.log1p(lastb / np.maximum(gapb, 0.5))
    D["buy_gap_mean"] = np.log1p(gapb)
    # кучность
    D["burstiness"] = (gsd - gm) / np.maximum(gsd + gm, 1e-6)
    D["fano"] = gsd ** 2 / np.maximum(gm, 1e-6)
    # агрегаты по последним K событиям, а не дням
    for k in (8, 16, 32, 64):
        sel = valid & (np.arange(K)[None, :] >= K - k)
        uu, ss = np.nonzero(sel)
        dd = DAYS[uu, ss]
        for c in ("gmv", "ord", "cart", "srch"):
            acc = np.zeros(NU)
            np.add.at(acc, uu, P[c][uu, dd])
            D[f"{c}_ev{k}"] = np.log1p(acc)
            D[f"{c}_per_ev{k}"] = np.log1p(acc / np.maximum(sel.sum(1), 1))
    # контрасты между событийными окнами
    for c in ("gmv", "cart"):
        D[f"{c}_ev_contrast"] = D[f"{c}_per_ev8"] - D[f"{c}_per_ev32"]
    D["span_ev64"] = np.log1p(np.where(valid.any(1), DAYS.max(1) - np.where(valid.any(1),
                              np.where(valid, DAYS, 10 ** 6).min(1), 0), a + 1).astype(np.float64))
    return D

def orth(d, basis):
    d = np.asarray(d, np.float64)
    d = d - d.mean()
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

print("=== 1. диагностика на локальном остатке (якорь 348) ===")
ref = OUT / "h8_gauss.npy"
if not ref.exists():
    sys.exit("нет локального предсказания h8_gauss.npy")
A0 = 348
p0 = np.load(ref)
ly0 = np.log1p(target(A0))
p0 = p0 - p0.mean() + ly0.mean()
res0 = ly0 - np.clip(p0, 0, 13)
qc0 = p0 - p0.mean()
one = np.ones(NU)
D0 = event_dirs(A0)
print(f"построено событийных направлений: {len(D0)}")
O0 = {}
for nm, v in D0.items():
    d = orth(v, [one, qc0])
    if d is not None:
        O0[nm] = d
rows = []
for nm, d in O0.items():
    c = float(np.corrcoef(res0, d)[0, 1])
    rows.append((abs(c * np.sqrt(NU - 2) / np.sqrt(max(1 - c * c, 1e-12))), nm, c))
print(f"\n{'направление':>20} {'corr':>10} {'t':>8}")
for t, nm, c in sorted(rows, reverse=True)[:14]:
    print(f"{nm:>20} {c:+10.5f} {t if c > 0 else -t:8.1f}")

names = list(O0)
Am = np.stack([O0[n] for n in names], 1)
lam = 1e-2 * len(names)
w = np.linalg.solve(Am.T @ Am + lam * np.eye(len(names)), Am.T @ res0)
comb0 = Am @ w
comb0 = comb0 / comb0.std()
c0 = float(np.corrcoef(res0, comb0)[0, 1])
print(f"\nсовместное событийное: corr {c0:+.5f}, t = {c0*np.sqrt(NU-2)/np.sqrt(max(1-c0*c0,1e-12)):.1f}")
print(f"  (для сравнения кликовое календарное давало corr +0.04746, t = 23.8)")
print(f"  топ весов: " + ", ".join(f"{n} {ww:+.3f}" for ww, n in
                                   sorted(zip(w, names), key=lambda x: -abs(x[0]))[:8]))
np.save(OUT / "j2_evdir_weights.npy", w)
open(OUT / "j2_evdir_names.txt", "w").write("\n".join(names))

print("\n=== 2. зонд на якоре 408, ортогонально к уже измеренному кликовому ===")
SRC = "submissions/stack_v14.csv"
R0 = 1.64577576410033
if not os.path.exists(SRC):
    SRC = "submissions/stack_v12.csv"; R0 = 1.646001202698559
    print(f"  stack_v14 не найден, опора {SRC}")
d0f = pl.read_csv(SRC).sort("user_id")
uid = d0f["user_id"].to_numpy()
q = np.log1p(d0f["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
# уже измеренное кликовое направление — обязательно в базис ортогонализации
prev_dir = None
pth = "submissions/probe_activity_comb.csv"
if os.path.exists(pth):
    qp = np.log1p(pl.read_csv(pth).sort("user_id")["predict"].to_numpy())
    qp = qp - qp.mean() + EL
    src12 = np.log1p(pl.read_csv("submissions/stack_v12.csv").sort("user_id")["predict"].to_numpy())
    src12 = src12 - src12.mean() + EL
    prev_dir = orth(qp - src12, [one, qc])
    print(f"  кликовое направление подхвачено, войдёт в базис ортогонализации")
D408 = event_dirs(408)
basis = [one, qc] + ([prev_dir] if prev_dir is not None else [])
comb = np.zeros(NU)
used = 0
for nm, wi in zip(names, w):
    if nm not in D408:
        continue
    dd = orth(D408[nm], basis)
    if dd is None:
        continue
    comb += wi * dd
    used += 1
comb = orth(comb, basis)
print(f"  собрано {used} направлений; corr с q {np.corrcoef(comb,q)[0,1]:+.2e}"
      + (f", corr с кликовым {np.corrcoef(comb,prev_dir)[0,1]:+.2e}" if prev_dir is not None else ""))
eps = float(np.sqrt((R0 + 0.0015) ** 2 - R0 ** 2))
def write_exact(name, t):
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
            lo = c
        else:
            hi = c
    pp = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    os.makedirs("submission/queue", exist_ok=True)
    pl.DataFrame({"user_id": uid, "predict": pp}).write_csv(f"submission/queue/{name}.csv")
    return float(np.log1p(pp).mean())
lv = write_exact("probe_event_norm", q + eps * comb)
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\nprobe_event_norm.csv")
print(f"  опора {R0:.7f}, eps {eps:.5f}, нуль {null:.6f}, цена {null-R0:.6f}, уровень {lv:.6f}")
print(f"  theta = (нуль^2 - R'^2) / (2*eps);  порог аппарата {R0*1e-4/eps:.4f}, "
      f"порог переноса {np.sqrt(R0**2/NPUB):.4f}")
print("done")
