"""Пересечение двух работающих осей: межклиентное сравнение НА КОНКРЕТНЫХ ОКНАХ.

Шесть замеров дали чёткий критерий: работает то, что НЕ ВЫВОДИТСЯ из собственного вектора
признаков клиента. Прошлогоднее окно (S/N 10.2) не выводится — этого куска истории нет ни в
одном признаке. Ранговая ось (S/N 10.2) не выводится — требует сравнения с другими клиентами.
А возраст, концентрация и стадия кривой ВЫВОДЯТСЯ из отношений окон, и дали ноль.

Здесь берётся пересечение двух работающих типов и то, что в j5 было утоплено смешиванием.

ЗОНД A — ранг во времени. Позиция клиента в популяции на конкретных окнах и траектория этой
позиции. Ранг сам по себе бесполезен (монотонное преобразование), полезны СДВИГИ и ФОРМА
траектории: они расходятся с абсолютной динамикой ровно тогда, когда двигалась вся популяция.
Ключевое: ранг в прошлогоднем целевом окне — пересечение обеих работающих осей.

ЗОНД B — когорта, изолированно. В j5 соседские сигналы шли вперемешку с выводимыми величинами
(возраст, концентрация) и, возможно, утонули в них: совместное направление собирается сложением
нормированных частей, поэтому шесть мёртвых частей разбавляют две живые втрое. Здесь только
межклиентное: поведение соседей в прошлогоднем окне, их траектория, однородность окрестности и
отклонение клиента от своей окрестности.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, NU, build_features
from sklearn.neighbors import NearestNeighbors

EL, VAR_L, NPUB = 2.3284, 5.366802, 50000
SRC = "submissions/stack_v17.csv"
R0 = 1.64550601334755
A = 408
YS, YE = 44, 73                      # прошлогоднее целевое окно
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)

def rank(x):
    r = np.empty(NU, np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(NU)
    return r / NU

def wsum(M, lo, hi):
    return M[:, max(0, lo):hi + 1].sum(1)

# ---------------- зонд A: ранг во времени
DA = {}
r_yago = rank(wsum(G, YS, YE))
r_now = rank(wsum(G, A - 29, A))
r_now90 = rank(wsum(G, A - 89, A))
DA["rank_yago"] = r_yago
DA["rank_now_minus_yago"] = r_now - r_yago
DA["rank_now90_minus_yago"] = r_now90 - r_yago
DA["rank_yago_vis"] = rank(wsum(V, YS, YE))
DA["rank_yago_vis_shift"] = rank(wsum(V, A - 29, A)) - rank(wsum(V, YS, YE))
# траектория ранга по шести блокам: наклон и волатильность
RB = np.stack([rank(wsum(G, A - 30 * (k + 1) + 1, A - 30 * k)) for k in range(6)], 1)
x = np.arange(6.0); x = (x - x.mean())
DA["rank_slope"] = (RB * x[None, ::-1]).sum(1) / (x ** 2).sum()
DA["rank_vol"] = RB.std(1)
DA["rank_last_minus_med"] = RB[:, 0] - np.median(RB, 1)
# условный ранг: позиция внутри собственного слоя активности
strat = np.clip((rank(wsum(V, A - 89, A)) * 10).astype(int), 0, 9)
cr = np.zeros(NU)
gv = wsum(G, A - 29, A)
for s in range(10):
    m = strat == s
    if m.sum() > 1:
        rr = np.empty(int(m.sum())); rr[np.argsort(gv[m], kind="stable")] = np.arange(int(m.sum()))
        cr[m] = rr / m.sum()
DA["cond_rank"] = cr
DA["cond_minus_global"] = cr - r_now

# ---------------- зонд B: когорта
print("строю признаки и ищу соседей ...", flush=True)
X, _ = build_features([A], verbose=False)
Z = X[A]
Zs = np.clip((Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-2), -8, 8).astype(np.float32)
top = np.argsort(np.var(Zs, 0))[::-1][:20]
nn = NearestNeighbors(n_neighbors=26, n_jobs=11).fit(Zs[:, top])
dist, idx = nn.kneighbors(Zs[:, top])
nb = idx[:, 1:]
del X, Z, nn
DB = {}
yl = np.log1p(wsum(G, YS, YE))
bg = np.log1p(wsum(G, max(0, YS - 45), YE + 45) / 105.0 * 30.0)
lift = yl - bg
DB["nb_yago_lift"] = lift[nb].mean(1)
DB["own_minus_nb_yago"] = lift - lift[nb].mean(1)
tr = r_now - rank(wsum(G, A - 59, A - 30))
DB["nb_trend"] = tr[nb].mean(1)
DB["own_minus_nb_trend"] = tr - tr[nb].mean(1)
DB["nb_dispersion"] = dist[:, 1:].mean(1)
DB["nb_rank_spread"] = r_now[nb].std(1)
lgnow = np.log1p(wsum(G, A - 29, A))
DB["own_minus_nb_level"] = lgnow - lgnow[nb].mean(1)
DB["nb_vis_trend"] = (rank(wsum(V, A - 29, A)) - rank(wsum(V, A - 59, A - 30)))[nb].mean(1)

# ---------------- общий аппарат
d0f = pl.read_csv(SRC).sort("user_id")
uid = d0f["user_id"].to_numpy()
q = np.log1p(d0f["predict"].to_numpy()); q = q - q.mean() + EL
qc = q - q.mean(); one = np.ones(NU)
basis = [one, qc]
PR = [("клик", "submissions/probe_activity_comb.csv", "submissions/stack_v12.csv"),
      ("событ", "submissions/probe_event_norm.csv", "submissions/stack_v14.csv"),
      ("прошлогод", "submissions/probe_yago.csv", "submissions/stack_v14.csv"),
      ("ранг", "submissions/probe_rank_shift.csv", "submissions/stack_v14.csv"),
      ("днед", "submissions/probe_dow_align.csv", "submissions/stack_v14.csv"),
      ("жизцикл", "submissions/probe_lifecycle.csv", "submissions/stack_v17.csv")]
for nm, path, base in PR:
    if os.path.exists(path) and os.path.exists(base):
        a_ = np.log1p(pl.read_csv(path).sort("user_id")["predict"].to_numpy())
        b_ = np.log1p(pl.read_csv(base).sort("user_id")["predict"].to_numpy())
        dd = (a_ - a_.mean()) - (b_ - b_.mean())
        for b in basis:
            dd = dd - np.dot(dd, b) / np.dot(b, b) * b
        if dd.std() > 1e-9:
            basis.append(dd / dd.std())
print(f"в базисе ортогонализации: {len(basis)-2} измеренных направлений")

def orth(d):
    d = np.nan_to_num(np.asarray(d, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    d = d - d.mean()
    for b in basis:
        nb_ = np.dot(b, b)
        if nb_ > 1e-12:
            d = d - np.dot(d, b) / nb_ * b
    s = d.std()
    return d / s if s > 1e-9 else None

eps = float(np.sqrt((R0 + 0.0015) ** 2 - R0 ** 2))
os.makedirs("submission/queue", exist_ok=True)

def emit(name, parts):
    print(f"\n{name}:")
    comb = np.zeros(NU); n = 0
    for nm, v in parts.items():
        dd = orth(v)
        if dd is None:
            print(f"  {nm:>24} вырождено"); continue
        print(f"  {nm:>24} corr с q {np.corrcoef(np.nan_to_num(v),q)[0,1]:+.4f}")
        comb += dd; n += 1
    comb = orth(comb)
    t = q + eps * comb
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL: lo = c
        else: hi = c
    p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(f"submission/queue/{name}.csv")
    mx = max(abs(np.corrcoef(comb, b)[0, 1]) for b in basis[2:])
    print(f"  из {n} частей, нуль {np.sqrt(R0**2+eps**2):.6f}, уровень {np.log1p(p).mean():.6f}, "
          f"макс corr с измеренными {mx:.1e}")

emit("probe_rank_time", DA)
emit("probe_cohort", DB)
print(f"\nопора {R0:.7f}, цена слота {np.sqrt(R0**2+eps**2)-R0:.6f}")
print(f"порог переноса |θ| > {np.sqrt(R0**2/NPUB):.4f}")
print("done")
