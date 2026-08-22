"""Изотоническая калибровка лестницы — на сохранённой матрице, без переобучения.

r2 показал систематический перебор: модель 0,5849 против факта 0,5623 на нижнем пороге,
0,5063 против 0,4836 в середине, сходится наверху. Тождество E[S|x] = sum_m h_m(x)*delta верно
ТОЛЬКО при калиброванных h_m.

Тонкость, которая определяет, стоит ли овчинка выделки. Если каждое h_m завышено на постоянную
delta, интеграл завышен на 33*delta*Delta — это КОНСТАНТА, и её съедает перекалибровка уровня.
Вредит только перекос ФОРМЫ. Поэтому меряем не «стало ли среднее ближе к факту», а изменился ли
скор после выравнивания уровня.

Честно: изотоника обучается на половине клиентов, применяется к другой, обе стороны.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from sklearn.isotonic import IsotonicRegression
from common import target, rmsle, NU, OUT

DELTA = 0.25
P = np.load(OUT / "r2_ladder_P.npy").astype(np.float64)
y = target(348); ly = np.log1p(y); MU = ly.mean()
GRID = np.arange(DELTA, DELTA * (P.shape[1] + 1) - 1e-9, DELTA)[:P.shape[1]]
print(f"матрица {P.shape}, порогов {len(GRID)}, верх {GRID[-1]:.2f}")

def sc(v, m=None):
    m = np.ones(NU, bool) if m is None else m
    return rmsle(y[m], np.expm1(np.clip(v[m] - v[m].mean() + ly[m].mean(), 0, 13)))

raw = DELTA * P.sum(1)
print(f"\nбез калибровки: {sc(raw):.5f}")

rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
cal = np.zeros_like(P)
for fm, em in ((half, ~half), (~half, half)):
    for j, s in enumerate(GRID):
        lab = (ly > s).astype(np.float64)
        if lab[fm].mean() < 1e-4 or lab[fm].mean() > 1 - 1e-4:
            cal[em, j] = P[em, j]; continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(P[fm, j], lab[fm])
        cal[em, j] = ir.predict(P[em, j])
iso = DELTA * cal.sum(1)
print(f"с изотоникой:   {sc(iso):.5f}   дельта {sc(iso)-sc(raw):+.2e}")

print(f"\nсдвиг калибровки по порогам (модель − факт):")
for j in range(0, len(GRID), 6):
    print(f"  s={GRID[j]:5.2f}: сырое {P[:,j].mean()-(ly>GRID[j]).mean():+.4f}, "
          f"после изотоники {cal[:,j].mean()-(ly>GRID[j]).mean():+.4f}")

print(f"\ncorr(сырое, изотоника) = {np.corrcoef(raw, iso)[0,1]:.6f}")
print(f"средние log1p: сырое {raw.mean():.4f}, изотоника {iso.mean():.4f}, факт {MU:.4f}")
np.save(OUT / "r9_ladder_iso.npy", iso.astype(np.float32))

# отдельно: помогает ли монотонизация лестницы по порогам
mono = DELTA * np.maximum.accumulate(P[:, ::-1], 1)[:, ::-1].sum(1)
print(f"\nмонотонизация без изотоники: {sc(mono):.5f}   дельта {sc(mono)-sc(raw):+.2e}")
print("\ndone")
