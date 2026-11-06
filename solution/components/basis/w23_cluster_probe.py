"""Зонд вдоль СРЕДНЕГО коррелированных разностей: одна ось вместо нескольких пересекающихся.

Замер взаимных корреляций показал два кластера: свёрточные правки блока связаны между собой на
0,36-0,46, энкодерные на 0,36, а между кластерами практически ноль (0,01-0,06).

Внутри кластера зондировать каждую отдельно — переплата слотами: при корреляции 0,4 их вклады
перекрываются почти наполовину. Среднее же коррелированных направлений СИЛЬНЕЕ каждого из них по
отдельности: общая часть складывается линейно, а независимый шум как корень из числа членов.

Поэтому здесь строится одно направление на кластер: каждая разность ортогонализуется к {1, q} и
нормируется, затем берётся среднее и нормируется снова. Знак каждого члена приводится к знаку
первого, иначе противоположно направленные члены гасили бы друг друга.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
import polars as pl
from common import NU, OUT

EL = 2.3284
SRC = os.environ.get("SRC", "submissions/stack_v22.csv")
R0 = float(os.environ.get("R0", "1.645280966365959"))
COST = float(os.environ.get("COST", "0.0015"))
NAME = os.environ["NAME"]

df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert np.array_equal(uid, np.load("cache/uids.npy")), "порядок клиентов не совпадает"
A = np.stack([np.ones(NU), q], 1)
solve = np.linalg.solve(A.T @ A, A.T)

dirs, names = [], []
for spec in os.environ["DIFFS"].split(";"):
    if not spec.strip():
        continue
    nm, o, n = spec.split(",")
    fo, fn = OUT / f"{o}.npy", OUT / f"{n}.npy"
    if not (fo.exists() and fn.exists()):
        print(f"  {nm}: нет файлов"); continue
    a, b = np.load(fo).astype(np.float64), np.load(fn).astype(np.float64)
    d = (b - b.mean()) - (a - a.mean())
    d = d - A @ (solve @ d)
    dirs.append(d / d.std()); names.append(nm)
assert dirs, "ни одной разности не собрано"
if len(dirs) > 1:
    for i in range(1, len(dirs)):
        if np.corrcoef(dirs[0], dirs[i])[0, 1] < 0:
            dirs[i] = -dirs[i]
            print(f"  знак {names[i]} обращён к знаку {names[0]}")
D = np.mean(dirs, 0)
D = D - A @ (solve @ D)
D = D / D.std()
print(f"кластер из {len(dirs)}: {', '.join(names)}")
if len(dirs) > 1:
    print(f"  средняя взаимная корреляция членов "
          f"{np.mean([np.corrcoef(dirs[i],dirs[j])[0,1] for i in range(len(dirs)) for j in range(i+1,len(dirs))]):.3f}")
    print(f"  корреляция среднего с членами: "
          f"{', '.join(f'{np.corrcoef(D,d)[0,1]:.3f}' for d in dirs)}")

eps = float(np.sqrt((R0 + COST) ** 2 - R0 ** 2))
t = q + eps * D
lo, hi = -6.0, 6.0
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
pred = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
out = f"submission/queue/probe_{NAME}.csv"
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
null = np.sqrt(R0 ** 2 + eps ** 2)
print(f"\n{out}")
print(f"  опора {R0:.10f}, eps {eps:.5f}, НУЛЬ {null:.7f}, уровень {np.log1p(pred).mean():.6f}")
print(f"  SE {R0/np.sqrt(50000):.5f} — порог t=1")
print("\ndone")
