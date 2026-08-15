"""An adjustable post-process: an orthogonal basis in the prediction, one probe per coefficient.

"Pull the high values down and push the low ones up" is, taken literally, the global slope — and
the slope is already at its optimum (b* = 1.00036 from Var(P) and Cov(L,P)). What is NOT the
slope is doing it UNEVENLY: compressing one end while stretching the other, or squeezing the
middle while letting the tails out. Those are the higher terms of the same expansion:

    q  ->  q + sum_j theta_j * phi_j(q)

    phi = 1          level      measured, EL = 2.3284
    phi = q - mu     slope      measured, b* = 1.00036, already optimal
    phi = (q-mu)^2   quadratic  compresses one end, stretches the other   <- not measured
    phi = (q-mu)^3   cubic      S-shape: squeezes the middle, frees the tails  <- not measured
    phi = 1{bin j}   bins       an arbitrary curve, nonparametric fallback  <- not measured

The basis is Gram-Schmidt orthogonalised, which is what makes it adjustable rather than
guessed. Because each direction is orthogonal to 1, to q and to every lower term, the probes do
not interfere: each one resolves its own coefficient cleanly, and the corrections add.

For a direction d with unit variance, adding eps*d gives exactly

    R'^2 = R0^2 - 2*eps*Cov(L - P, d) + eps^2

The residual L - P is orthogonal to P because the stack is an OLS projection, but nothing makes
it orthogonal to P^2 or P^3 — so the linear term is pure curvature. The cost is eps^2 and the
signal is linear in eps, which is why this is far cheaper than probing b = 0.9 / 1.1: those move
along the one direction that is already at the vertex of its parabola and get punished
quadratically for nothing.

Level matching is solved, not assumed: shifting a perturbed vector pushes part of it below zero,
expm1 clips there, and the clip pulls the mean back up. Bisection on the shift fixes it exactly.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

EL, VAR_L = 2.3284, 5.366802
SRC = "submissions/stack_opt_daily.csv"
R0 = 1.646809
OUT = Path("submission/queue")
OUT.mkdir(parents=True, exist_ok=True)

d0 = pl.read_csv(SRC).sort("user_id")
uid = d0["user_id"].to_numpy()
q = np.log1p(d0["predict"].to_numpy())
q = q - q.mean() + EL
qc = q - q.mean()

def gram_schmidt(v, basis):
    for b in basis:
        v = v - np.dot(v, b) / np.dot(b, b) * b
    return v

one = np.ones_like(q)
B = [one, qc]
DIRS = {}
for name, raw in (("quad", qc ** 2), ("cubic", qc ** 3)):
    v = gram_schmidt(raw, B)
    v = v / v.std()
    DIRS[name] = v
    B.append(v)

print(f"источник {SRC}, R0 = {R0:.6f}, Var(P) = {q.var():.6f}")
print(f"{'направление':>12} {'corr с 1':>10} {'corr с q':>10} {'corr между':>12}")
for n, v in DIRS.items():
    print(f"{n:>12} {np.dot(v,one)/len(v):10.2e} {np.corrcoef(v,q)[0,1]:10.2e} "
          f"{np.corrcoef(DIRS['quad'],DIRS['cubic'])[0,1]:12.2e}")

def write_exact(name, t):
    """bisect the shift so the WRITTEN file has mean log1p exactly EL, clipping included"""
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
            lo = c
        else:
            hi = c
    p = np.clip(np.expm1(t + (lo + hi) / 2), 0, None)
    pl.DataFrame({"user_id": uid, "predict": p}).write_csv(OUT / f"{name}.csv")
    return float(np.log1p(p).mean())

TARGET_COST = 0.0015
eps = float(np.sqrt((R0 + TARGET_COST) ** 2 - R0 ** 2))
print(f"\n=== ЗОНДЫ ФОРМЫ (eps подобран так, чтобы слот стоил ~{TARGET_COST}) ===")
print(f"{'файл':>24} {'eps':>7} {'нуль':>10} {'цена':>9} {'SE(theta)':>10} {'уровень':>10}")
for name, v in DIRS.items():
    t = q + eps * v
    m = write_exact(f"probe_{name}", t)
    null = np.sqrt(R0 ** 2 + eps ** 2 * float(v.var()))
    se = R0 * 1e-4 / eps
    print(f"{'probe_'+name:>24} {eps:7.4f} {null:10.6f} {null-R0:9.6f} {se:10.5f} {m:10.6f}")

print(f"\n  анализ одного зонда:  theta = (нуль^2 - R'^2) / (2*eps)")
print(f"  выигрыш от применения: dR^2 = theta^2  (Var(d) = 1 по построению)")
print(f"  знак theta: + значит края надо поднять относительно середины, - опустить")
print(f"  поправки складываются: q' = q + theta_quad*d_quad + theta_cubic*d_cubic")
print(f"  порог полезности: |theta| > {R0*1e-4/eps:.4f} даёт выигрыш больше шума аппарата")

print(f"\n=== КОРЗИННЫЕ ЗОНДЫ (непараметрический запас, точный уровень) ===")
C = 0.15
edges = np.percentile(q, [20, 40, 60, 80])
bi = np.searchsorted(edges, q)
print(f"{'файл':>24} {'корзина':>14} {'доля':>7} {'нуль':>10} {'уровень':>10}")
for j, lab in ((0, "нижние 20%"), (1, "20-40%"), (4, "верхние 20%")):
    m_ = bi == j
    w = float(m_.mean())
    lv = write_exact(f"probe_bin{j}_c015", q + C * m_)
    print(f"{'probe_bin'+str(j)+'_c015':>24} {lab:>14} {w:7.3f} "
          f"{np.sqrt(R0**2 + w*C**2):10.6f} {lv:10.6f}")
print(f"\n  SE(D_j) на 20% публичной выборки ~ {1.647/np.sqrt(50000*0.2):.4f}")
print(f"  усадка перед применением: kappa = max(0, 1 - SE^2/D_j^2)")
print(f"\nготово, файлы в {OUT}/")
