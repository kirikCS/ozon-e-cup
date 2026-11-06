"""Усреднение реализаций внутри половин пары: одна старая версия, одна новая.

Зачем усреднять именно так. Замер по четырём сидам показал, что вклад ОДНОЙ реализации гуляет
впятеро при корреляции векторов 0,9986, а среднее по сидам даёт вклад ЛУЧШЕ любой отдельной
реализации. Значит и половины пары надо брать усреднёнными, иначе разность будет измерять в
основном разницу двух случайных прогонов.

Разности по отдельным сидам это подтверждают: 0,00475 / 0,00150 / 0,00329 / 0,00309, то есть
разброс втрое. Честная величина эффекта эпох — около 0,0032, а не 0,0047, как выходило по
первому сиду.

Здесь строятся четыре файла: усреднённые старая и новая половины на каждом протоколе.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, NU, OUT

y = target(348); ly = np.log1p(y)
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))

GROUPS = {
    "x16_ev_old": ["v13_ep7", "v13_ep7s1", "v13_ep7s2", "v13_ep7s3"],
    "x16_ev_new": ["v13_ep20", "v13_ep20s1", "v13_ep20s2", "v13_ep20s3"],
    "x16_dl_old": ["x10_nll", "x10s1_nll"],
    "x16_dl_new": ["x10_e2e", "x10s1_e2e"],
}
for out, members in GROUPS.items():
    for proto in ("sel", "fin"):
        vs, got = [], []
        for m in members:
            f = OUT / f"{m}_{proto}.npy"
            if f.exists():
                vs.append(np.load(f).astype(np.float64)); got.append(m)
        if not vs:
            print(f"{out}_{proto}: НЕТ НИ ОДНОГО файла из {members}")
            continue
        E = np.mean(vs, 0)
        np.save(OUT / f"{out}_{proto}.npy", E.astype(np.float32))
        msg = f"{sc(E):.5f}" if proto == "sel" else "—"
        cs = ""
        if len(vs) > 1:
            c = [np.corrcoef(vs[i], vs[j])[0, 1] for i in range(len(vs)) for j in range(i+1, len(vs))]
            cs = f", взаимная corr {np.mean(c):.5f}"
        print(f"{out}_{proto}: усреднено {len(vs)} из {len(members)} ({', '.join(got)}), "
              f"RMSLE {msg}{cs}")

print("\nразности на SELECT:")
for tag, o, n in (("событийная", "x16_ev_old", "x16_ev_new"), ("дневная", "x16_dl_old", "x16_dl_new")):
    fo, fn = OUT / f"{o}_sel.npy", OUT / f"{n}_sel.npy"
    if fo.exists() and fn.exists():
        a, b = np.load(fo).astype(np.float64), np.load(fn).astype(np.float64)
        print(f"  {tag}: {sc(a):.5f} -> {sc(b):.5f}, дельта {sc(b)-sc(a):+.5f}, "
              f"sd разности {((b-b.mean())-(a-a.mean())).std():.5f}")
print("\ndone")
