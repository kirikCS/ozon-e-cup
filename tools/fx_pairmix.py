"""Смеси ДВУХ измеренных векторов: один подгоняемый параметр вместо двадцати девяти.

Ключ, которого не хватало: шумовой потолок public->private растёт как sqrt(d),
где d - число ПОДГОНЯЕМЫХ параметров, а не радиус смещения. Нулевой тест мерил
d=20 и дал потолок 3.1e-4 на радиусе 0.027. Для ОДНОГО параметра тот же потолок
равен 3.1e-4/sqrt(20) = 7.0e-5.

Значит выпуклая смесь двух ИЗМЕРЕННЫХ векторов a*X + (1-a)*Y - это подгонка одного
числа, и её публичный выигрыш переживает приват, если превышает ~1.4e-4 (S/N 2).
Ровно та конструкция, что разрешена финальной политикой проекта (text.md:511-514):
каждая компонента измерена на LB собственным R*, никаких лонг-шортов.

Для пары измеренных векторов алгебра ТОЧНАЯ и модель-свободная:
    m(a)^2 = a^2 mX^2 + (1-a)^2 mY^2 + 2a(1-a) * Cov(eX, eY)
где Cov(eX,eY) выводится из третьего замера или из ковариаций. Здесь используем
базисный аппарат (k у измеренных векторов точен) - но проверяем результат
модель-свободной формулой через скалярное произведение ошибок.

Usage: python -u code/fx_pairmix.py > model_out/pairmix.log 2>&1
"""
import itertools
import numpy as np
import polars as pl
from pathlib import Path

EL, VAR_L = 2.3284, 5.366802
NULL_27 = 0.000313          # шумовой потолок при d=20, радиус 0.027
FLOOR_1P = NULL_27 / 20**0.5

MEAS = [
    ("РЕКОРД", "submissions/kirill_stack_final_lb1646487.csv", 1.646487285085104),
    ("blend_opt", "submissions/measured/blend_opt_lb1646684.csv", 1.6466839752880942),
    ("ensemble_v3", "submissions/measured/ensemble_v3_lb1646702.csv", 1.6467023706525434),
    ("opt_daily", "submissions/kirill_stack_opt_daily_lb1646812.csv", 1.6468118221271069),
    ("fusion_v2", "submissions/measured/fusion_v2_lb1646856.csv", 1.6468560110518367),
    ("lgbm_v5", "submissions/lgbm_blend_v5.csv", 1.6473863695245514),
    ("stack13", "submissions/stack13_lb1647414.csv", 1.6474141758705303),
    ("blend_v4", "submissions/twink_blend_v4_lb1647997.csv", 1.6479965555704743),
    ("unc", "submissions/kirill_unc_lb1648469.csv", 1.648469231589679),
    ("dl_fusion", "submissions/stack_dl_fusion_lb1648049.csv", 1.6480488273),
]

Q, uid = {}, None
for n, p, lb in MEAS:
    d = pl.read_csv(p)
    o = np.argsort(d["user_id"].to_numpy())
    if uid is None: uid = d["user_id"].to_numpy()[o]
    q = np.log1p(d["predict"].to_numpy()[o])
    Q[n] = q - q.mean() + EL
LBm = {n: lb for n, _, lb in MEAS}
REC = LBm["РЕКОРД"]

# Cov(eX, eY) выводится модель-свободно:
#   m_X^2 = Var(eX), m_Y^2 = Var(eY) (уровни выровнены)
#   Var(X - Y) = Var(eX - eY) = m_X^2 + m_Y^2 - 2Cov(eX,eY)
#   => Cov(eX,eY) = (m_X^2 + m_Y^2 - Var(X-Y)) / 2
# Var(X-Y) считается прямо по файлам. Никакого k, никакой подгонки.
print("=" * 88)
print("ВЫПУКЛЫЕ СМЕСИ ДВУХ ИЗМЕРЕННЫХ ВЕКТОРОВ (один подгоняемый параметр)")
print("=" * 88)
print(f"шумовой потолок для d=1: {FLOOR_1P:.6f}  (против {NULL_27:.6f} при d=20)")
print(f"порог доверия S/N=2: выигрыш должен быть > {2*FLOOR_1P:.6f}")
print()
print(f"{'пара':>26} {'a*':>7} {'m(a*)':>11} {'выигрыш':>10} {'S/N':>6} {'вердикт':>12}")
rows = []
for X, Y in itertools.combinations(Q, 2):
    mX2, mY2 = LBm[X]**2, LBm[Y]**2
    vd = float(np.var(Q[X] - Q[Y]))
    cxy = (mX2 + mY2 - vd) / 2
    den = mX2 + mY2 - 2*cxy
    if den <= 1e-12:
        continue
    a = (mY2 - cxy) / den                      # argmin по a
    a = float(np.clip(a, 0.0, 1.0))
    m2 = a*a*mX2 + (1-a)**2*mY2 + 2*a*(1-a)*cxy
    m = float(np.sqrt(max(m2, 0)))
    g = REC - m
    sn = g / FLOOR_1P
    if g <= 0:
        continue
    rows.append((g, sn, X, Y, a, m))
rows.sort(reverse=True)
for g, sn, X, Y, a, m in rows[:14]:
    v = "НАСТОЯЩИЙ" if sn > 2 else "спорный" if sn > 1 else "подгонка"
    print(f"{X+' / '+Y:>26} {a:7.3f} {m:11.7f} {g:+10.6f} {sn:6.2f} {v:>12}")

if not rows:
    print("  ни одна пара не бьёт рекорд")
else:
    g, sn, X, Y, a, m = rows[0]
    print()
    print("=" * 88)
    print(f"ЛУЧШАЯ ПАРА: {a:.4f}*{X} + {1-a:.4f}*{Y}")
    print("=" * 88)
    mix = a*Q[X] + (1-a)*Q[Y]
    mix = mix + (EL - mix.mean())
    pout = np.clip(np.expm1(mix), 0, None)
    fn = Path("submissions/PAIR_best.csv")
    with open(fn, "w") as f:
        f.write("user_id,predict\n")
        for u, val in zip(uid, pout):
            f.write(f"{u},{val:.6f}\n")
    d2 = pl.read_csv(fn)
    q2 = np.log1p(d2["predict"].to_numpy()[np.argsort(d2["user_id"].to_numpy())])
    q2 = q2 - q2.mean() + EL
    print(f"  прогноз (модель-свободный) {m:.7f}   выигрыш {g:+.6f}   S/N {sn:.2f}")
    print(f"  ЗАПИСАН {fn}")
    print(f"  строк {len(pout)}  нулей {100*float((pout<=0).mean()):.2f}%  "
          f"mean_lp {np.log1p(pout).mean():.5f}")
    print(f"  1-corr с рекордом "
          f"{(1-float(np.corrcoef(q2, Q['РЕКОРД'])[0,1]))*1e6:.0f}e-6")
    print(f"  веса обеих компонент в [0,1], лонг-шорта нет -> проходит финальную")
    print(f"  политику text.md:511-514")
    print()
    print("Прогноз получен БЕЗ вектора k: только измеренные lb двух файлов и")
    print("Var(X-Y), посчитанная прямо по csv. Модельного риска здесь нет вообще.")
