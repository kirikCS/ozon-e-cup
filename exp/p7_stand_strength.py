"""Тает ли вклад квантильных рук при усилении стенда.

p6 дал -3,95e-4 на стенде из 20 векторов уровня 1,7349. Настоящий стек - 33+ вектора на 1,6455.
Выигрыши на слабом стенде обычно тают на сильном, поэтому смотрим зависимость вклада от силы:
стенд из k лучших векторов, k = 3, 5, 10, 15, 20. Если вклад держится - можно нести в финал.
"""
import sys; sys.path.insert(0, "code")
import numpy as np
from pathlib import Path
from common import target, rmsle, OUT

y = target(348); ly = np.log1p(y)
BASE, CAND = {}, {}
for p in sorted(Path(OUT).glob("*.npy")):
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (250000,) or not np.isfinite(v).all(): continue
    s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    if s >= 1.80: continue
    (CAND if p.stem.startswith("p5_") else BASE)[p.stem] = (s, v - v.mean())
order = sorted(BASE, key=lambda k: BASE[k][0])
q99 = [CAND[k][1] for k in CAND if k.endswith("_q99")]

def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(), 1))], 1)
    G = A.T @ A + lam*len(A)*np.eye(A.shape[1]); G[-1, -1] -= lam*len(A)
    w = np.linalg.solve(G, A.T @ ly[fm])
    v = M @ w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em], 0, 13)))

rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
SPL = ((half, ~half), (~half, half))
print(f"{'векторов':>9} {'стенд':>10} {'+обе q99':>11} {'вклад':>11}")
for k in (3, 5, 10, 15, 20):
    cols = [BASE[n][1] for n in order[:k]]
    b  = np.mean([fit_eval(cols, f, e) for f, e in SPL])
    bq = np.mean([fit_eval(cols + q99, f, e) for f, e in SPL])
    print(f"{k:9d} {b:10.6f} {bq:11.6f} {bq-b:+11.2e}", flush=True)
print("\ndone")
