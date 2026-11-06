"""Сборка стека с заменёнными компонентами: направление-РАЗНОСТЬ с весом компоненты.

Замена c_j на улучшенную c_j' меняет прогноз стека на w_j * (c_j' - c_j). Три отличия от зонда,
который мы уже потратили:
  строится РАЗНОСТЬ старой и новой версии, а не сама новая версия;
  добавляется с ФИКСИРОВАННЫМ весом компоненты, а не с подобранным лидербордом;
  проекция на текущий прогноз НЕ вычитается — улучшение той части, что уже в стеке, это и есть
  главный смысл замены, а зонд её явно выбрасывал.

Веса берутся с якоря 348: ридж по базису, куда добавлены СТАРЫЕ половины пар, и вес старой
половины — это и есть w_j. Лидерборд про веса ничего сказать не может, поэтому здесь только он и
подбирается локально; сама разность на финальном якоре берётся как есть.

Обе половины каждой пары посчитаны одним прогоном при одном сиде: измерено, что две реализации
одной конфигурации различаются по вкладу вдвое, и из разных прогонов эта разница уехала бы прямо
в разность.

Уровень файла калибруется на EL = 2,3284 двоичным поиском, как во всех наших сабмитах.
"""
import sys, os
sys.path.insert(0, "code")
import re as _re
import numpy as np
import polars as pl
from pathlib import Path
from common import target, rmsle, NU, OUT

CANDRX = _re.compile(r"^[suvwx]\d+_")
EL = 2.3284
SRC = "submissions/stack_v21.csv"
LEAK = 1.70
# пары задаются переменной: событийная готова (4 сида), дневная ждёт машину.
# Собирать по одной паре за раз — тогда замер лидербордом однозначен: при смеси двух
# изменений разной природы непонятно, какая половина сработала, а какая помешала.
_P = {"event": ("x16_ev_old", "x16_ev_new"), "daily": ("x16_dl_old", "x16_dl_new")}
PAIRS = [(k, *_P[k]) for k in os.environ.get("PAIRS", "event").split(",")]
SCALE = float(os.environ.get("SCALE", 1.0))     # общий множитель на осторожность

y = target(348); ly = np.log1p(y)
BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (NU,) or not np.isfinite(v).all():
        continue
    sc = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if sc >= 1.80 or "fin" in p.stem or sc < LEAK or CANDRX.match(p.stem):
        continue
    BASE[p.stem] = v - v.mean()
print(f"в базисе {len(BASE)}", flush=True)

old_sel, cols, names = [], list(BASE.values()), list(BASE)
for tag, o, n in PAIRS:
    f = OUT / f"{o}_sel.npy"
    if not f.exists():
        print(f"НЕТ {f.name} — пара {tag} не готова"); sys.exit(1)
    v = np.load(f).astype(np.float64)
    print(f"  {tag}: старая половина {o}_sel, RMSLE "
          f"{rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13))):.5f}")
    cols.append(v - v.mean()); names.append(o)

M = np.stack(cols, 1)
A = np.concatenate([M, np.ones((NU, 1))], 1)
lam = 1e-4
G = A.T @ A + lam * NU * np.eye(A.shape[1]); G[-1, -1] -= lam * NU
w = np.linalg.solve(G, A.T @ ly)
W = {names[i]: float(w[i]) for i in range(len(names))}
print("\nвеса старых половин в ридже:")
for tag, o, n in PAIRS:
    print(f"  {tag:>6} ({o}): {W[o]:+.5f}")

df = pl.read_csv(SRC)
uid = df["user_id"].to_numpy()
q = np.log1p(df["predict"].to_numpy().astype(np.float64))
assert np.array_equal(uid, np.load("cache/uids.npy")), "порядок клиентов не совпадает"

delta = np.zeros(NU)
for tag, o, n in PAIRS:
    fo, fn = OUT / f"{o}_fin.npy", OUT / f"{n}_fin.npy"
    if not (fo.exists() and fn.exists()):
        print(f"НЕТ финальной пары для {tag}"); sys.exit(1)
    a_, b_ = np.load(fo).astype(np.float64), np.load(fn).astype(np.float64)
    d = (b_ - b_.mean()) - (a_ - a_.mean())
    print(f"  {tag}: sd разности {d.std():.5f}, corr половин {np.corrcoef(a_, b_)[0,1]:.5f}")
    delta += W[o] * d
delta *= SCALE
print(f"\nсуммарный сдвиг: sd {delta.std():.5f}, corr с прогнозом {np.corrcoef(delta, q)[0,1]:+.4f}")

t = q + delta
lo, hi = -6.0, 6.0
for _ in range(90):
    c = (lo + hi) / 2
    if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
        lo = c
    else:
        hi = c
pred = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
os.makedirs("submission/queue", exist_ok=True)
out = "submission/queue/stack_v22_" + "_".join(p[0] for p in PAIRS) + ".csv"
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(out)
print(f"\n{out}")
print(f"  уровень {np.log1p(pred).mean():.6f}, corr с v21 {np.corrcoef(np.log1p(pred), q)[0,1]:.6f}")
print(f"  среднее |сдвига| в логарифме: {np.abs(delta).mean():.5f}")
print("\nэто НЕ зонд: файл собран как готовая замена, мерить надо прямо")
print("\ndone")
