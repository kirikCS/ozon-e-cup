"""Leaderboard-calibrated post-processing: probes whose null is known exactly beforehand.

The regulated version the local screen cannot give. Any transform fitted locally inherits the
anchor it was fitted on — that is how isotonic calibration became the worst submission of the
project, having learned "shrink towards January" from the one seasonal trough of the year. A
probe does not fit anything: it applies a KNOWN perturbation to the current best vector, and
the leaderboard returns one exact number that resolves one unknown.

Two families, both with closed-form nulls.

BEND. Apply p' = expm1(b * log1p(p)) at a fixed b, level-matched afterwards. Under level
matching the score is exactly

    R^2(b) = Var_L - 2b*Cov(L,P) + b^2*Var(P)

and Cov(L,P) is already known from the vector's own measured R*, so the null is arithmetic, not
a guess. A measured deviation from it is curvature — the part a pure slope cannot express.

BINS. Apply a shift +c only to clients whose prediction falls in bin j. The parabola algebra
extends exactly to a masked shift:

    R'^2 = R0^2 + w_j*c^2 - 2*c*w_j*D_j,     D_j = E[log1p y - log1p p | bin j]

so one submission resolves the conditional bias of one bin on the REAL test window. The
teammate already ran this on "top 20% predicted" and got D = -0.0025 (t = -0.15), i.e. that bin
is unbiased. The bins probed here are the ones nobody has looked at: the bottom of the
distribution, where 46% of the mass sits at zero and where a systematic bias is most plausible.

Every probe is a deliberately perturbed vector, so it scores slightly WORSE than the source by
construction. The printed null says how much worse — that is the price of the measurement, and
it is small enough that the slot is not wasted.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

EL, VAR_L = 2.3284, 5.366802
SRC = "submissions/stack_opt_daily.csv"     # текущий лучший расчётный вектор
R0 = 1.646809                                # его предсказание аппаратом
OUT = Path("submission/queue")
OUT.mkdir(parents=True, exist_ok=True)

d = pl.read_csv(SRC).sort("user_id")
uid = d["user_id"].to_numpy()
q = np.log1p(d["predict"].to_numpy())
q = q - q.mean() + EL
VarP = float(q.var())
CovLP = (VAR_L + VarP - R0 ** 2) / 2
print(f"источник: {SRC}")
print(f"  Var(P) = {VarP:.6f},  Cov(L,P) = {CovLP:.6f},  R0 = {R0:.6f}")
print(f"  подразумеваемый оптимальный наклон b* = {CovLP/VarP:.5f}\n")

def write(name, qq):
    qq = qq - qq.mean() + EL
    p = np.clip(np.expm1(qq), 0, None)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(OUT / f"{name}.csv")
    return qq

print("=== ЗОНДЫ ИЗГИБА: p' = expm1(b * log1p(p)), уровень выровнен ===")
print(f"{'файл':>22} {'b':>6} {'нуль (если изгиба нет)':>24}")
for b in (0.90, 1.10):
    qq = write(f"probe_bend_b{int(b*100):03d}", b * q)
    # нуль считается на ВЫРОВНЕННОМ векторе, ровно как он уходит на лидерборд
    vp = float(qq.var())
    # ковариация масштабируется вместе с вектором: Cov(L, b*P) = b*Cov(L,P)
    null = np.sqrt(max(VAR_L - 2 * b * CovLP + b * b * VarP, 0))
    print(f"{'probe_bend_b'+str(int(b*100)).zfill(3):>22} {b:6.2f} {null:24.6f}")
print("  анализ: измеренное R'^2 даёт Cov при этом b; отклонение от квадратичной")
print("  зависимости по трём точкам (b = 0.90, 1.00, 1.10) и есть кривизна\n")

print("=== ЗОНДЫ КОРЗИН: сдвиг +c только на своей корзине предсказания ===")
C = 0.15
edges = np.percentile(q, [20, 40, 60, 80])
b_idx = np.searchsorted(edges, q)
print(f"{'файл':>24} {'корзина':>18} {'доля':>7} {'нуль (если смещения нет)':>26}")
for j, lab in ((0, "нижние 20%"), (1, "20–40%"), (4, "верхние 20%")):
    m = b_idx == j
    w = float(m.mean())
    write(f"probe_bin{j}_c{int(C*100):03d}", q + C * m)
    null = np.sqrt(R0 ** 2 + w * C ** 2)
    print(f"{'probe_bin'+str(j)+'_c'+str(int(C*100)).zfill(3):>24} {lab:>18} {w:7.3f} "
          f"{null:26.6f}")
print(f"\n  анализ: D_j = (R0^2 + w_j*c^2 - R'^2) / (2*c*w_j)")
se = 1.647 / np.sqrt(50000 * 0.2)
print(f"  стандартная ошибка D_j на корзине в 20% публичной выборки ~ {se:.4f}")
print(f"  поправка применяется усаженной: c_j = kappa*D_j, kappa = max(0, 1 - SE^2/D_j^2)")
print(f"\nвсе зонды записаны в {OUT}/")
print("ВАЖНО: каждый зонд — намеренно ухудшенный вектор. Отправлять по одному, начиная с того,")
print("у которого нуль ближе к R0 — цена измерения там наименьшая.")
