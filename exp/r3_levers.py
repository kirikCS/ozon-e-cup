"""Иерархия рычагов в ОСТАТКАХ, а не в предсказаниях.

Всё это время g мерилась как корреляция векторов предсказаний, и вся иерархия сжалась в полосу
0,980-0,999. Порог допуска corr(L,новая) > corr(L,стек)*g от этого не ломается — он выводится
именно через корреляции предсказаний. Но РАЗРЕШЕНИЕ шкалы теряется: рычаги, различающиеся втрое
по реальной пользе, выглядят как 0,995 против 0,997.

Величина выигрыша двухкомпонентной смеси управляется корреляцией остатков rho:
    при равной силе компонент  MSE_смеси / MSE = (1+rho)/2,  выигрыш RMSLE = 1 - sqrt((1+rho)/2)
Поэтому здесь считаются обе величины рядом, и к каждой паре приписан ожидаемый выигрыш.
"""
import sys; sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

y = target(348); ly = np.log1p(y); MU = ly.mean()
def load(n):
    p = Path(OUT)/f"{n}.npy"
    if not p.exists(): return None
    v = np.load(p).astype(np.float64)
    return v - v.mean() + MU
def sc(v): return rmsle(y, np.expm1(np.clip(v, 0, 13)))

PAIRS = [
    ("сид (та же конфигурация)",      "h8_gauss",      "n4_gen"),
    ("функция потерь (гаусс/лаплас)", "h8_gauss",      "h8_laplace"),
    ("функция потерь (гаусс/стьюдент)","h8_gauss",     "h8_student"),
    ("праздничные признаки",          "p2_plain",      "p2_hol"),
    ("архитектура (tft/свёртка)",     "h8_gauss",      "d2_cnn_sel"),
    ("ствол (cnn/слияние)",           "p5_cnn_gen",    "p5_fusion_gen"),
    ("ГОЛОВА (генератив/квантиль)",   "p5_cnn_gen",    "p5_cnn_q99"),
    ("голова, второй ствол",          "p5_fusion_gen", "p5_fusion_q99"),
    ("ствол+голова вместе",           "p5_cnn_gen",    "p5_fusion_q99"),
    ("семейство (свёртка/CoLES)",     "h8_gauss",      "g5_coles_grn_sel"),
    ("тип модели (нейро/бустинг)",    "h8_gauss",      "r1_l2_log1p"),
    ("тип модели (нейро/kNN)",        "h8_gauss",      "i2_knn_k200"),
]
print(f"{'рычаг':>32} {'corr предск.':>13} {'corr остатк.':>13} {'ожид. выигрыш':>14}")
rows = []
for nm, a, b in PAIRS:
    va, vb = load(a), load(b)
    if va is None or vb is None:
        print(f"{nm:>32} {'нет вектора':>13} {'('+a if va is None else '('+b}) ")
        continue
    ra, rb = ly - va, ly - vb
    cp = np.corrcoef(va, vb)[0, 1]
    cr = np.corrcoef(ra, rb)[0, 1]
    # выигрыш оптимальной смеси двух компонент неравной силы
    sa, sb = ra.std(), rb.std(); k = sb/sa
    ratio = k*k*(1-cr*cr) / (1 - 2*cr*k + k*k)
    gain = (1 - np.sqrt(ratio)) * sa / (2*rmsle(y, np.expm1(np.clip(va,0,13)))) * 2*np.sqrt(ratio)
    # проще и точнее: посчитать напрямую
    w = (sa*sa - cr*sa*sb) / (sa*sa - 2*cr*sa*sb + sb*sb)
    mix = w*va + (1-w)*vb
    g2 = sc(va) - sc(mix - mix.mean() + MU)
    rows.append((nm, cp, cr, g2, sc(va), sc(vb)))
    print(f"{nm:>32} {cp:13.5f} {cr:13.5f} {g2:+14.5f}")
print(f"\n{'':>32} {'':>13} {'':>13} {'выигрыш = сколько даёт оптимальная смесь пары'}")
print("\nдля справки, скоры компонент:")
for nm, cp, cr, g2, s1, s2 in rows:
    print(f"  {nm:>32}: {s1:.5f} / {s2:.5f}")
print("\ndone")
