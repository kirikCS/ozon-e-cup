"""Четвёртый набор внешних осей: жизненный цикл клиента и поведение его когорты.

Критерий, выведенный из пяти замеров, теперь двойной:
  1. величины нет в фичесторе КАК КЛАССА (прошлогоднее окно прошло, S/N 10.2);
  2. величина влияет на МЕСЯЧНЫЙ ИТОГ, а не только на дневной рисунок.

Второе условие добавлено после провала выравнивания по дню недели (t = +0.60): тридцатидневная
сумма содержит ~4.3 недели и усредняет недельную структуру, перекос в один день это 3% состава.

Что берётся:

TENURE — сколько клиент вообще существует. Фичестор содержит СЧЁТЧИКИ (buyday_all, ord_all), но
не возраст: клиент с 20 покупками за два месяца и клиент с 20 покупками за год — разные, а
счётчики у них одинаковые. Влияет на месячный итог напрямую через стадию жизненного цикла.

RAMP — на какой стадии кривой клиент. Активность первых 30 дней жизни против последних 30:
новички разгоняются, старики затухают, и скорость этого не выводится из счётчиков.

CONCENTRATION — доля пожизненных трат в последних 90 днях. Отличает клиента, который всегда
тратил ровно, от того, кто весь объём сделал недавно, при равных суммах.

COHORT — поведение соседей. Средняя динамика ранга у ближайших по признакам клиентов. Это
единственная величина здесь, требующая МЕЖКЛИЕНТНОГО сравнения: ни одна компонента не смотрит
на других клиентов при предсказании данного. kNN как модель провалился (отставание 0.025), но
как направление это другой объект — мы берём не его предсказание, а сигнал о когорте.
"""
import sys, os
import numpy as np
import polars as pl

sys.path.insert(0, "code")
from common import panel, NU, build_features

EL, VAR_L, NPUB = 2.3284, 5.366802, 50000
SRC = "submissions/stack_v17.csv"
R0 = 1.64550601334755
A = 408
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
O = panel("ord").astype(np.float32)

def lg(x):
    return np.log1p(np.maximum(x, 0))

D = {}
actm = V[:, :A + 1] > 0
buym = G[:, :A + 1] > 0
first_act = np.argmax(actm, 1).astype(np.float64)
first_act[~actm.any(1)] = A
first_buy = np.argmax(buym, 1).astype(np.float64)
first_buy[~buym.any(1)] = A
D["tenure_act"] = lg(A - first_act)
D["tenure_buy"] = lg(A - first_buy)
D["tenure_gap"] = lg(first_buy - first_act)          # сколько ходил до первой покупки

# стадия кривой: первые 30 дней жизни против последних 30
ui = np.arange(NU)
f0 = first_act.astype(int)
early = np.zeros(NU, np.float64)
for off in range(30):
    idx = np.clip(f0 + off, 0, A)
    early += G[ui, idx]
late = G[:, A - 29:A + 1].sum(1)
D["ramp"] = lg(late) - lg(early)
D["ramp_vis"] = lg(V[:, A - 29:A + 1].sum(1)) - lg(np.take_along_axis(
    V, np.clip(f0[:, None] + np.arange(30)[None, :], 0, A), 1).sum(1))

# концентрация: доля пожизненного объёма в последних 90 днях
tot = G[:, :A + 1].sum(1)
D["conc_90"] = G[:, A - 89:A + 1].sum(1) / np.maximum(tot, 1e-6)
D["conc_30"] = G[:, A - 29:A + 1].sum(1) / np.maximum(tot, 1e-6)
D["conc_norm"] = D["conc_90"] - 90.0 / np.maximum(A - first_act, 1)

# когорта: динамика ранга у ближайших соседей по признакам
print("строю признаки для поиска соседей ...", flush=True)
X, names = build_features([A], verbose=False)
Z = X[A]
mu, sd = Z.mean(0), np.maximum(Z.std(0), 1e-2)
Zs = np.clip((Z - mu) / sd, -8, 8).astype(np.float32)
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
top = np.argsort(np.var(Zs, 0))[::-1][:20]
def rank(x):
    r = np.empty(NU, np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(NU)
    return r / NU
own = rank(G[:, A - 29:A + 1].sum(1)) - rank(G[:, A - 59:A - 29].sum(1))
print("ищу соседей ...", flush=True)
nn = NearestNeighbors(n_neighbors=26, n_jobs=11).fit(Zs[:, top])
_, idx = nn.kneighbors(Zs[:, top])
D["cohort_drank"] = own[idx[:, 1:]].mean(1)
D["cohort_minus_own"] = D["cohort_drank"] - own
del X, Z, Zs, nn, idx

d0f = pl.read_csv(SRC).sort("user_id")
uid = d0f["user_id"].to_numpy()
q = np.log1p(d0f["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()
one = np.ones(NU)
basis = [one, qc]
PR = [("клик", "submissions/probe_activity_comb.csv", "submissions/stack_v12.csv"),
      ("событ", "submissions/probe_event_norm.csv", "submissions/stack_v14.csv"),
      ("прошлогод", "submissions/probe_yago.csv", "submissions/stack_v14.csv"),
      ("ранг", "submissions/probe_rank_shift.csv", "submissions/stack_v14.csv"),
      ("днед", "submissions/probe_dow_align.csv", "submissions/stack_v14.csv")]
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
    d = np.asarray(d, np.float64)
    d = d - d.mean()
    for b in basis:
        nb = np.dot(b, b)
        if nb > 1e-12:
            d = d - np.dot(d, b) / nb * b
    s = d.std()
    return d / s if s > 1e-9 else None

print(f"\n{'величина':>18} {'corr с q':>11} {'остаётся после ортог.':>22}")
comb = np.zeros(NU); n = 0
for nm, v in D.items():
    v = np.nan_to_num(np.asarray(v, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    raw_sd = v.std()
    dd = orth(v)
    if dd is None:
        print(f"{nm:>18} {'вырождено':>11}")
        continue
    keep = np.std(v - v.mean() - np.sum([np.dot(v - v.mean(), b) / np.dot(b, b) * b
                                         for b in basis], 0)) / max(raw_sd, 1e-9)
    print(f"{nm:>18} {np.corrcoef(v,q)[0,1]:+11.4f} {keep:22.3f}")
    comb += dd; n += 1
comb = orth(comb)
mx = max(abs(np.corrcoef(comb, b)[0, 1]) for b in basis[2:]) if len(basis) > 2 else 0.0
print(f"\nсовместное из {n}: corr с q {np.corrcoef(comb,q)[0,1]:+.1e}, "
      f"макс corr с измеренными {mx:.1e}")

eps = float(np.sqrt((R0 + 0.0015) ** 2 - R0 ** 2))
t = q + eps * comb
lo, hi = -6.0, 6.0
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL: lo = c
    else: hi = c
p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
pl.DataFrame({"user_id": uid, "predict": p}).write_csv("submission/queue/probe_lifecycle.csv")
print(f"\nprobe_lifecycle.csv: опора {R0:.7f}, нуль {np.sqrt(R0**2+eps**2):.6f}, "
      f"цена {np.sqrt(R0**2+eps**2)-R0:.6f}, уровень {np.log1p(p).mean():.6f}")
print(f"порог аппарата |θ| > {R0*1e-4/eps:.4f}, порог переноса |θ| > {np.sqrt(R0**2/NPUB):.4f}")
print("done")
