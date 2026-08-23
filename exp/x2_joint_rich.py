"""Та же диагностика, но против базовой линии, которую наша модель РЕАЛЬНО знает.

Дыра в x1: базовая линия там — плоский темп клиента, умноженный на общий календарь дня. Но наша
модель видит 364 дня истории и умеет больше: локальный уровень (клиент разогнался или затухает) и
собственный недельный ритм. Всё это внутри горизонта выглядит как зависимость между днями, хотя
на деле предсказуемо и уже учтено в p_d.

Поэтому здесь базовая линия богатая и строго ПРИЧИННАЯ:
  уровень   — скользящее среднее за предыдущие 28 дней, только прошлое;
  ритм      — собственный недельный множитель клиента, оценённый на днях 60..259 и применённый
              к дням 260..408, то есть на непересекающемся куске;
  календарь — общий множитель дня, как и раньше.

Что остаётся после такой линии, модель предсказать НЕ МОЖЕТ даже в принципе: все 30 дней горизонта
считаются из одной истории, и реализованная покупка 349-го дня недоступна при предсказании 350-го.

Сравнение двух профилей и есть ответ. Если богатая линия убирает большую часть превышения —
структура предсказуема, и поправлять считыватель незачем. Если превышение сохраняется — оно
настоящее, и двухуровневая frailty в квадратуре оправдана.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
from common import panel, D0 as DATE0
from datetime import timedelta

W, FIT_HI, EV_LO, EV_HI = 28, 260, 260, 409
LAGS = list(range(1, 15)) + [21, 28]
G = panel("gmv")
Z = (G > 0).astype(np.float32)
T = Z.shape[1]
wd = np.array([(DATE0 + timedelta(days=int(d))).weekday() for d in range(T)])
print(f"панель {Z.shape}, доля активных {Z.mean():.4f}", flush=True)

# 1. собственный недельный ритм: оценка на 60..259, применение на 260..408
fit = Z[:, 60:FIT_HI]
wdf = wd[60:FIT_HI]
mult = np.ones((Z.shape[0], 7), np.float32)
base = fit.mean(1, keepdims=True) + 1e-6
for k in range(7):
    sel = wdf == k
    mult[:, k] = fit[:, sel].mean(1) / base[:, 0]
mult = np.clip(mult, 0.2, 5.0)

# 2. причинный локальный уровень: среднее за ПРЕДЫДУЩИЕ W дней
cs = np.cumsum(Z, axis=1, dtype=np.float32)
lvl = np.zeros_like(Z)
lvl[:, W:] = (cs[:, W:] - cs[:, :-W]) / W
lvl[:, :W] = Z[:, :W].mean(1, keepdims=True)

ev = slice(EV_LO, EV_HI)
Ze = Z[:, ev]
E = lvl[:, ev] * mult[:, wd[ev]]
dayadj = Ze.mean(0) / np.maximum(E.mean(0), 1e-9)
E = E * dayadj[None, :]
E = np.clip(E, 1e-5, 0.999).astype(np.float32)
R = (Ze - E).astype(np.float32)
print(f"богатая линия: средняя p {E.mean():.4f}, факт {Ze.mean():.4f}", flush=True)

# плоская линия на том же куске — для честного сравнения
r = Ze.mean(1, keepdims=True); m = Ze.mean(0, keepdims=True)
Rf = (Ze - r * m / Ze.mean()).astype(np.float32)

def prof(X, lags):
    sd = X.std()
    return [float((X[:, :-k] * X[:, k:]).mean() / (sd * sd)) for k in lags]

pr, pf = prof(R, LAGS), prof(Rf, LAGS)
rng = np.random.default_rng(0)
idx = np.argsort(rng.random(Ze.shape), axis=1)
Zs = np.take_along_axis(Ze, idx, axis=1)
rs = Zs.mean(1, keepdims=True); ms = Zs.mean(0, keepdims=True)
pc = prof((Zs - rs * ms / Zs.mean()).astype(np.float32), LAGS)

print(f"\n{'лаг':>5} {'плоская линия':>15} {'БОГАТАЯ линия':>15} {'контроль':>11} {'осталось':>11}")
for k, a, b, c in zip(LAGS, pf, pr, pc):
    print(f"{k:>5} {a-c:15.5f} {b-c:15.5f} {c:11.5f} {(b-c)/max(a-c,1e-9):11.2f}")
print("""
Последняя колонка — доля превышения, ПЕРЕЖИВШАЯ богатую линию. Около нуля значит, что структура
предсказуема из истории и уже сидит в p_d. Близко к единице значит, что она принципиально
недоступна модели, которая считает все дни горизонта из одной истории.""")
print("\ndone")
