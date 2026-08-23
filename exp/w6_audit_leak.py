"""Аудит утечки: какие векторы в model_out знают цель якоря 348.

Стенд оценивает на якоре 348, цель — сумма GMV за дни 349-378. Вектор, построенный на ФИНАЛЬНОМ
якоре 408, читает историю по 408-й день включительно, то есть дни 349-378 лежат у него прямо
НА ВХОДЕ. Его предсказание — функция от целевого окна. Это прямая утечка, и такой вектор нельзя
пускать ни в кандидаты, ни в базис.

Обнаруживается двумя независимыми способами:
  по имени   — подстрока "fin";
  по числу   — одиночный RMSLE на 348 заметно ниже всего, что достижимо честно. Лучшая наша
               одиночная модель — лестница порогов, 1,73556; дневная 1,741. Порог 1,70 с запасом.

Второй способ важнее первого: имя можно и забыть проставить.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

y = target(348); ly = np.log1p(y)
rows = []
for p in sorted(Path(OUT).glob("*.npy")):
    try:
        v = np.load(p)
    except Exception:
        continue
    if v.shape != (250000,) or not np.isfinite(v).all():
        continue
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    rows.append((s, p.stem))
rows.sort()
print(f"{'вектор':>28} {'RMSLE на 348':>13}   пометка")
for s, n in rows[:14]:
    mark = ""
    if s < 1.70:
        mark = "УТЕЧКА по числу"
    if "fin" in n:
        mark = (mark + " и по имени").strip() if mark else "по имени 'fin', но число честное"
    print(f"{n:>28} {s:13.5f}   {mark}")
print(f"\nвсего векторов {len(rows)}")
bad = [n for s, n in rows if s < 1.70 or "fin" in n]
print(f"под подозрением {len(bad)}: {bad}")
print("\ndone")
