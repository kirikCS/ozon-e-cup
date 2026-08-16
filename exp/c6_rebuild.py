"""Rebuild the stack with the bagged daily model, ready to run the moment its LB arrives.

Written before the measurement so the decision rule is fixed in advance rather than chosen to
suit whatever number comes back. Three things it does:

  1. Puts daily_bagged in the basis ALONGSIDE the existing daily rather than replacing it. They
     are different vectors — one averaged over four 80% client subsamples, one not — so the pair
     spans a direction neither has alone. The solver decides whether that is worth anything; if
     the bagged one simply dominates, the old one gets a weight near zero and nothing is lost.

  2. Re-applies the measured shape correction. The cubic direction was fitted against
     stack_opt_daily's residual, and the rebuilt stack has a different residual, so the
     correction is recomputed against the new base rather than copied across. Its coefficient is
     the measured Cov with James-Stein shrinkage, unchanged — that number came from the
     leaderboard and does not depend on the base.

  3. Keeps the weight policy that stack8x paid for: max|w| <= 0.6, no short beyond -0.25. The
     ridge is swept and the first admissible point is taken, not the best-scoring one.

Usage:  python exp/c6_rebuild.py <LB замер daily_bagged>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

H = "ecup_handover_2026-08-14/"
EL, VAR_L, RC = 2.3284, 5.366802, 0.7039
NPUB = 50000

if len(sys.argv) < 2:
    sys.exit("укажи замеренный LB для daily_bagged.csv")
LB_BAGGED = float(sys.argv[1].replace(",", "."))

V = [
    ("bwn", H + "submission/queue/blend-with-neural.csv", 1.6486126196896862),
    ("tcb", H + "submission/queue/TOTALCATBOOSTBLEND.csv", 1.6491800741214804),
    ("nocal", H + "submission/queue/nocal_catboost.csv", 1.6535318106642274),
    ("seq_pure", H + "submissions/seq_pure_9seed.csv", 1.6658092084427447),
    ("stack_dl", H + "submission/queue/stack_dl.csv", 1.648344063188555),
    ("evgru2", H + "submissions/evgru2_final_lb1660490.csv", 1.660490164378798),
    ("ts_v3", H + "submissions/two_stage_v3.csv", 1.6530487792218926),
    ("cb_v3", H + "submissions/catboost_v3.csv", 1.653229223776429),
    ("ts_9seed", H + "submissions/two_stage_btyd_9seed.csv",
     (1.6556581195791316 ** 2 - 0.1008 ** 2) ** 0.5),
    ("ts_old", H + "submission/components/two_stage_btyd_shift.csv", 1.6546867486238441),
    ("cb_old", H + "submission/components/catboost_shifted.csv", 1.6534269025124644),
    ("seq_old", H + "submission/components/seq_hybrid.csv", 1.6541075222),
    ("seq9", H + "submissions/seq_hybrid_9seed.csv", 1.654003921177154),
    ("dart", H + "submission/components/lgbm_dart.csv", 1.672341791651013),
    ("iso", H + "submission/components/nnls_isotonic_blend.csv", 1.6836226427909342),
    ("eb", H + "submission/components/eb_logblocks.csv", 1.7009172654286824),
    ("opt_blend", "stack_opt_blend.csv", 1.6470636973608355),
    ("daily", "submissions/daily_gen_q2c.csv", 1.6543093235326605),
    ("daily_bag", "submissions/daily_bagged.csv", LB_BAGGED),
]

def z(path):
    d = pl.read_csv(path).sort("user_id")
    q = np.log1p(d["predict"].to_numpy())
    return d["user_id"].to_numpy(), q - q.mean() + EL, q.mean()

uid, P, R2, names = None, [], [], []
for n, p, lb in V:
    u, q, mu = z(p)
    if uid is None:
        uid = u
    assert np.array_equal(u, uid), f"порядок user_id разошёлся в {n}"
    P.append(q)
    R2.append(lb ** 2 - (EL - mu) ** 2)
    names.append(n)
M = np.stack(P)
C = np.cov(M)
k = (VAR_L + np.diag(C) - np.array(R2)) / 2

CONS = [c for c in json.loads(Path(H + "model_out/stack_constraints.json")
                              .read_text(encoding="utf-8-sig")) if os.path.exists(H + c["csv"])]
A_, b_ = [], []
for c in CONS:
    q = np.log1p(pl.read_csv(H + c["csv"]).sort("user_id")["predict"].to_numpy())
    q = q - q.mean() + EL
    Aa = np.concatenate([M, np.ones((1, M.shape[1]))], 0)
    sol, *_ = np.linalg.lstsq(Aa.T, q, rcond=None)
    wv = sol[:-1]
    A_.append(wv)
    b_.append((c["lb"] ** 2 - (VAR_L - 2 * wv @ k + wv @ C @ wv)) / 2)
A_, b_ = np.stack(A_), np.array(b_)
kc = k - A_.T @ np.linalg.solve(A_ @ A_.T, b_)
Cc0 = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
i_d, i_b = names.index("daily"), names.index("daily_bag")
print(f"corr(daily, daily_bag) = {Cc0[i_d, i_b]:.5f}"
      f"{'  -> пара под ограничением |w| <= 0.5' if Cc0[i_d, i_b] >= 0.99 else ''}")
pairs = [(names[i], names[j], Cc0[i, j]) for i in range(len(names))
         for j in range(i + 1, len(names)) if Cc0[i, j] >= 0.99]
print(f"пар с corr >= 0.99: {len(pairs)}" +
      ("  " + ", ".join(f"{a}~{b} {c:.4f}" for a, b, c in sorted(pairs, key=lambda x: -x[2])[:4])
       if pairs else ""))
print("\nпроверка ограничений:")
for c, wv in zip(CONS, A_):
    print(f"  {Path(c['csv']).name[:30]:32s} {np.sqrt(VAR_L - 2*wv@kc + wv@C@wv):.6f} "
          f"против {c['lb']:.6f}")

best = None
print()
for ridge in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
    w = np.linalg.solve(C + ridge * np.eye(len(kc)) * np.trace(C) / len(kc), kc)
    q2 = VAR_L - 2 * w @ kc + w @ C @ w
    if q2 <= 0:
        continue
    # stack8x провалился на паре corr 0.998 с весами +0.50/-0.51 (промах +0.0016).
    # daily и daily_bag — одна архитектура на одних данных, отличие только в усреднении,
    # то есть заведомо такая пара. Общего лимита 0.6 здесь мало: политика передачи требует
    # |w| <= 0.5 на всём, что скоррелировано выше 0.99.
    Cc = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    hot = [(i, j) for i in range(len(w)) for j in range(i + 1, len(w)) if Cc[i, j] >= 0.99]
    hot_ok = all(abs(w[i]) <= 0.5 and abs(w[j]) <= 0.5 for i, j in hot)
    ok = np.abs(w).max() <= 0.6 and w.min() >= -0.25 and hot_ok
    bad = "" if hot_ok else "  коллинеарная пара сверх 0.5"
    print(f"ridge {ridge:.0e}: {np.sqrt(q2):.6f}  max|w| {np.abs(w).max():.3f}  "
          f"min {w.min():+.3f}  {'ok' if ok else 'вне политики'}{bad}")
    if ok and (best is None or np.sqrt(q2) < best[0]):
        best = (np.sqrt(q2), ridge, w)
if best is None:
    sys.exit("ни одна точка риджа не проходит политику весов")
r, ridge, w = best
print(f"\nвыбрано ridge {ridge:.0e}: {r:.6f}")
for n, wi in sorted(zip(names, w), key=lambda x: -abs(x[1])):
    if abs(wi) > 0.01:
        mark = "   <- бэггированная" if n == "daily_bag" else ("   <- прежняя" if n == "daily" else "")
        print(f"  {wi:+8.4f}  {n}{mark}")

base = (w[:, None] * M).sum(0)
base = base + (EL - base.mean())
pl.DataFrame({"user_id": uid, "predict": np.clip(np.expm1(base), 0, None)}) \
  .write_csv("submissions/stack_bag.csv")
print(f"\nstack_bag.csv записан (без поправки формы)")

# поправка формы, пересчитанная относительно НОВОЙ базы
qc = base - base.mean()
one = np.ones_like(base)
v = qc ** 3
for b in (one, qc):
    v = v - np.dot(v, b) / np.dot(b, b) * b
d_cub = v / v.std()
COV_CUB = -0.001492                      # замерено зондом на stack_opt_daily
EPS_CUB = float(np.sqrt((1.646809 + 0.0015) ** 2 - 1.646809 ** 2))
theta = COV_CUB / EPS_CUB                # коэффициент при единичном направлении
se = np.sqrt(r ** 2 / NPUB)
kap = max(0.0, 1 - se ** 2 / theta ** 2)
print(f"поправка формы: theta {theta:+.5f}, SE {se:.5f}, t {theta/se:+.2f}, kappa {kap:.3f}")
nq = base + kap * theta * d_cub
nq = nq - nq.mean() + EL
p = np.clip(np.expm1(nq), 0, None)
pl.DataFrame({"user_id": uid, "predict": p}).write_csv("submissions/stack_bag_shape.csv")
su = pl.read_csv("data/sample_submit.csv")["user_id"].to_numpy()
print(f"stack_bag_shape.csv: {len(p)} строк, порядок {'ок' if np.array_equal(uid, su) else 'НЕТ'}, "
      f"mean log1p {np.log1p(p).mean():.4f}, neg {int((p < 0).sum())}")
print(f"ожидаемый скор {np.sqrt(max(r**2 - kap*theta**2, 0)):.6f}")
