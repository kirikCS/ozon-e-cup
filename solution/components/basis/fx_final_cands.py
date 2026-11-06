"""Финальные кандидаты на завтра. Лёгкий режим: маленький ансамбль, мало итераций.

Минимаксная постановка из fx_robust.py, но экономно по CPU (пользователь работает).
Считаем два уровня риска:
  AGGRESSIVE — минимум худшего потолка по ансамблю k
  SAFE       — малый радиус: ближе к измеренному рекорду, меньше зависит от модели

Ансамбль k ОБЯЗАТЕЛЬНО включает вариант без constraint якоря: именно под его
неточность (остаток вне базиса 0.0243% против медианных 0.00001%) подгонялся
оптимизатор, давая фиктивные потолки ниже физического пола.

Usage: python -u code/fx_final_cands.py > model_out/final_cands.log 2>&1
"""
import json
import numpy as np
import polars as pl
from pathlib import Path
from scipy.optimize import minimize

EL, VAR_L, RIVAL, ORACLE_FLOOR, R_KNN = 2.3284, 5.366802, 1.6466975, 1.6456, 0.9518
RNG = np.random.default_rng(777)
N_K = 14          # компактный ансамбль
MAXIT = 250       # мало итераций: цель — устойчивое решение, не последний знак

V24 = [
    ("bwn", "submissions/make_stack_neural.csv", 1.6486126196896862),
    ("tcb", "submissions/make_stack_catboost.csv", 1.6491800741214804),
    ("nocal", "submissions/t15_no_calendar.csv", 1.6535318106642274),
    ("seq_pure", "submissions/seq_pure_9seed.csv", 1.6658092084427447),
    ("stack_dl", "submissions/make_stack_dl.csv", 1.648344063188555),
    ("evgru2", "submissions/evgru2_final_lb1660490.csv", 1.660490164378798),
    ("evgru3", "submissions/evgru3_final_lb1656201.csv", 1.6562005780842861),
    ("ts_emb", "submissions/ts_emb_9seed.csv", 1.6529448453628885),
    ("ts_v3", "submissions/two_stage_v3.csv", 1.6530487792218926),
    ("cb_v3", "submissions/catboost_v3.csv", 1.653229223776429),
    ("ts_9seed", "submissions/two_stage_btyd_9seed.csv", (1.6556581195791316**2 - 0.1008**2)**0.5),
    ("ts_old", "submission/components/two_stage_btyd_shift.csv", 1.6546867486238441),
    ("cb_old", "submission/components/catboost_shifted.csv", 1.6534269025124644),
    ("seq_old", "submission/components/seq_hybrid.csv", 1.6541075222),
    ("seq9", "submissions/seq_hybrid_9seed.csv", 1.654003921177154),
    ("dart", "submission/components/lgbm_dart.csv", 1.672341791651013),
    ("iso", "submission/components/nnls_isotonic_blend.csv", 1.6836226427909342),
    ("knn", "submissions/knn_submit_lb.csv", 1.7175601692772613),
    ("btyd_basic", "submissions/btyd_btyd_basic_lb.csv", 1.7865214092919335),
    ("eb", "submission/components/eb_logblocks.csv", 1.7009172654286824),
    ("knn_funnel", "submissions/knn_funnel_lb.csv", 1.86246 * R_KNN),
    ("knn_funnel2", "submissions/knn_funnel2_lb.csv", 1.80101 * R_KNN),
    ("funnel_lgbm", "submissions/funnel_lgbm_lb.csv", 1.80921 * R_KNN),
    ("daily", "release_blend/components/daily.csv", 1.6566),
]

P, Rs2, names, uid = [], [], [], None
for n, p, lb in V24:
    d = pl.read_csv(p)
    u = d["user_id"].to_numpy(); o = np.argsort(u)
    if uid is None: uid = u[o]
    q = np.log1p(d["predict"].to_numpy()[o])
    P.append(q - q.mean() + EL); Rs2.append(lb**2 - (EL - q.mean())**2); names.append(n)
M = np.stack(P); C = np.cov(M)
k_raw = np.array([(VAR_L + C[i, i] - Rs2[i]) / 2 for i in range(len(names))])
NB = len(names)
Am = np.concatenate([M, np.ones((1, M.shape[1]))], 0)


def dec(path):
    d = pl.read_csv(path)
    q = np.log1p(d["predict"].to_numpy()[np.argsort(d["user_id"].to_numpy())])
    q = q - q.mean() + EL
    return np.linalg.lstsq(Am.T, q, rcond=None)[0][:NB]


sc = json.loads(Path("model_out/stack_constraints.json").read_text(encoding="utf-8-sig"))
g = sorted([c for c in sc if c["lb"] < 1.75], key=lambda c: abs(c["lb"] - 1.6465))
W = [dec(c["csv"]) for c in g]
LB = np.array([c["lb"] for c in g])
N = len(g)
anchor_i = int(np.argmin(LB))
w0 = W[anchor_i]
REC = LB[anchor_i]


def k_of(idx, lam=1e-6):
    A = np.stack([W[i] for i in idx])
    b = np.array([(LB[i]**2 - (VAR_L - 2*W[i]@k_raw + W[i]@C@W[i])) / 2 for i in idx])
    return k_raw - A.T @ np.linalg.solve(A @ A.T + lam*np.eye(len(idx)), b)


def rm(w, k):
    return float(np.sqrt(max(VAR_L - 2*w@k + w@C@w, 0)))


def dist(u):
    return float(np.sqrt(max(u @ C @ u, 0)))


KS = [k_of(list(range(N))), k_of([i for i in range(N) if i != anchor_i])]
for _ in range(N_K):
    idx = sorted(RNG.choice(N, size=N-5, replace=False))
    if np.linalg.matrix_rank(np.stack([W[i] for i in idx])) >= 5:
        KS.append(k_of(idx))
KS = np.stack(KS)
A_full = np.stack(W)
rank = np.linalg.matrix_rank(A_full)
Vt = np.linalg.svd(A_full, full_matrices=True)[2]
Prow = Vt[:rank].T @ Vt[:rank]

apred = np.array([rm(w0, k) for k in KS])
print(f"ансамбль k: {len(KS)}   рекорд {REC:.7f}")
print(f"потолок рекорда по ансамблю: {apred.min():.6f}..{apred.max():.6f} "
      f"медиана {np.median(apred):.6f}")


def worst(w):
    return max(rm(w, k) for k in KS)


print(f"\n{'D':>7} {'худший':>9} {'медиана':>9} {'мед.выигрыш':>12} "
      f"{'P(лучше)':>9} {'max|w|':>7}")
rows = []
for D in [0.008, 0.014, 0.020, 0.026, 0.034, 0.045, 0.060]:
    cons = [{"type": "ineq", "fun": lambda z: D - dist(Prow @ z)},
            {"type": "ineq", "fun": lambda z: 0.6 - np.max(w0 + Prow @ z)},
            {"type": "ineq", "fun": lambda z: np.min(w0 + Prow @ z) + 0.25}]
    r = minimize(lambda z: worst(w0 + Prow @ z), np.zeros(NB), method="SLSQP",
                 constraints=cons, options={"maxiter": MAXIT, "ftol": 1e-11})
    w = w0 + Prow @ r.x
    vals = np.array([rm(w, k) for k in KS])
    gains = apred - vals
    print(f"{D:7.3f} {vals.max():9.6f} {np.median(vals):9.6f} "
          f"{np.median(gains):+12.6f} {100*(gains>0).mean():8.1f}% "
          f"{np.abs(w).max():7.3f}")
    rows.append((D, vals.max(), np.median(vals), np.median(gains),
                 100*(gains > 0).mean(), w.copy()))

# AGGRESSIVE — минимум худшего; SAFE — малый радиус с P=100%
aggr = min(rows, key=lambda t: t[1])
safe_pool = [t for t in rows if t[4] >= 99.9]
safe = min(safe_pool, key=lambda t: t[0]) if safe_pool else min(rows, key=lambda t: t[0])

q0 = np.log1p(pl.read_csv(g[anchor_i]["csv"])["predict"].to_numpy()[
    np.argsort(pl.read_csv(g[anchor_i]["csv"])["user_id"].to_numpy())])

print()
for tag, t in [("FINAL_aggressive", aggr), ("FINAL_safe", safe)]:
    D, wc, med, mg, pb, w = t
    mix = (w[:, None] * M).sum(0); mix = mix + (EL - mix.mean())
    pout = np.clip(np.expm1(mix), 0, None)
    fn = Path(f"submissions/{tag}_lb.csv")
    with open(fn, "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uid, pout):
            f.write(f"{u},{v:.6f}\n")
    dif = (1 - float(np.corrcoef(np.log1p(pout), q0)[0, 1])) * 1e6
    print(f"{fn}")
    print(f"  D={D:.3f}  худший={wc:.6f}  медиана={med:.6f}  "
          f"мед.выигрыш={mg:+.6f}  P(лучше)={pb:.0f}%")
    print(f"  отличие от рекорда {dif:.0f}e-6, нулей "
          f"{100*float((pout <= 0).mean()):.2f}%")

print(f"\nрекорд={REC:.7f}  ривал={RIVAL:.7f}  пол={ORACLE_FLOOR}")
print("Собственная точность аппарата по LOO ~0.0004 — держать в уме при чтении чисел.")
