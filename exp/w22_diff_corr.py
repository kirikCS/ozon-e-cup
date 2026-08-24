"""Взаимная зависимость направлений-разностей: какие зондировать, а какие дубликаты.

Выигрыш складывается в квадратуре только у НЕЗАВИСИМЫХ направлений. Пять правок свёрточного блока
меняют одну и ту же модель, и их разности почти наверняка связаны — тогда их theta не сложатся,
а перекроются, и слоты уйдут впустую.

Здесь каждая разность строится так же, как в зонде (ортогонализация к {1, q} опоры, нормировка на
единичный разброс) и считаются взаимные корреляции УЖЕ ОРТОГОНАЛИЗОВАННЫХ направлений — именно они
определяют, сложатся вклады или нет.

Порог для отбора: при корреляции 0,7 совместный вклад двух направлений примерно на четверть меньше
суммы; при 0,9 — вдвое. Брать в зонд стоит наименее связанные.
"""
import sys, os
sys.path.insert(0, "code")
import numpy as np
import polars as pl
from pathlib import Path
from common import NU, OUT

SRC = os.environ.get("SRC", "submissions/stack_v21.csv")
df = pl.read_csv(SRC)
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
A = np.stack([np.ones(NU), q], 1)
AtA_inv_At = np.linalg.solve(A.T @ A, A.T)

def direction(old, new):
    fo, fn = OUT / f"{old}.npy", OUT / f"{new}.npy"
    if not (fo.exists() and fn.exists()):
        return None
    a = np.load(fo).astype(np.float64); b = np.load(fn).astype(np.float64)
    d = (b - b.mean()) - (a - a.mean())
    d = d - A @ (AtA_inv_At @ d)
    return d / d.std()

PAIRS = {}
for spec in os.environ.get("DIFFS", "").split(";"):
    if not spec.strip():
        continue
    nm, o, n = spec.split(",")
    PAIRS[nm] = (o, n)
if not PAIRS:
    PAIRS = {
        "оператор":  ("x16_ev_old_fin", "v12_declm1_fin"),
        "окно":      ("x16_ev_old_fin", "v8_deep_fin"),
        "эпохи":     ("x16_ev_old_fin", "x16_ev_new_fin"),
        "вентиль":   ("y1_base_fin", "y1_gated_fin"),
        "instnorm":  ("y1_base_fin", "y1_instnorm_fin"),
        "se":        ("y1_base_fin", "y1_se_fin"),
        "bigk":      ("y1_base_fin", "y1_bigk_fin"),
        "rope":      ("y2_base_fin", "y2_rope_fin"),
        "reg":       ("y2_base_fin", "y2_reg_fin"),
        "цель":      ("x16_dl_old_fin", "x16_dl_new_fin"),
    }
D, names = {}, []
for nm, (o, n) in PAIRS.items():
    d = direction(o, n)
    if d is None:
        print(f"{nm:>10}: нет файлов ({o} / {n})")
        continue
    D[nm] = d; names.append(nm)
print(f"\nготовых направлений: {len(names)}")
if len(names) > 1:
    print("\nвзаимные корреляции ортогонализованных направлений:")
    print("           " + " ".join(f"{k:>10}" for k in names))
    for a_ in names:
        print(f"{a_:>10} " + " ".join(f"{np.corrcoef(D[a_], D[b])[0,1]:10.3f}" for b in names))
    print("\nчем ближе к нулю, тем лучше складываются вклады")
print("\ndone")
