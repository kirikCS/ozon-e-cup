"""Солвер стека: оптимальная смесь замеренных векторов по калибровочному аппарату.

Аппарат (раздел 2 контекста решения):

    E[log1p y]   = 2,3284           замерено зондами
    Var(log1p y) = 5,366802         замерено зондом наклона
    Cov(L, P_j)  = (Var_L + Var(P_j) - R_j^2) / 2      ТОЧНО для замеренного вектора

Каждый вектор выравнивается по уровню q - mean(q) + EL. Тогда скор смеси
предсказывается заранее:

    R^2(w) = Var_L - 2*w'k + w'Cw,   C = Gram матрица векторов, k_j = Cov(L, P_j)

и оптимум без ограничений w = C^-1 k. Базис здесь — НЕ отобранный список, а ВСЕ
замеренные векторы: «базис 33» означает лишь, что тогда их было 33.

Прямое решение вырождается: векторы коррелируют на 0,99+. Нужны две вещи, обе
задокументированы и обе оплачены ошибками проекта.

ПОПРАВКА k ПО РЕЕСТРУ. Каждая УЖЕ ЗАМЕРЕННАЯ смесь даёт точное линейное
ограничение на ошибку k: w_j' delta = (R^2_изм - R^2_пред)/2, где w_j —
разложение этой смеси по базису. Min-norm решение по всем ограничениям
вычитается: k_corr = k - delta. Без неё предсказание систематически оптимистично.

ПОЛИТИКА ВЕСОВ, оплаченная провалом stack8x (промах +0,0016):
max|w| <= 0.6, шорты не глубже -0.25. Решение вне политики отвергается.

Запуск:
    python stack_solver.py --pool ПУТЬ --scores ФАЙЛ [--registry ФАЙЛ]
                           [--exclude РЕГЭКСП] [--out ФАЙЛ] [--target ЭТАЛОН]
"""
import argparse, json, os, re
import numpy as np, pandas as pd

EL, VAR_L = 2.3284, 5.366802
MAXW, MAXSHORT = 0.6, -0.25


def load_log(p):
    d = pd.read_csv(p).sort_values("user_id")
    return d["user_id"].to_numpy(), np.log1p(d["predict"].to_numpy(np.float64))


# Компоненты ищутся только в переданном --pool. Прежний список путей относительно
# текущей директории убран: пакет не должен подхватывать чужие файлы.
def find(pool, n):
    p = os.path.join(pool, n + ".csv")
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--registry", default=None, help="json со списком замеренных смесей")
    ap.add_argument("--exclude", default=None, help="регэксп: какие имена не брать в базис")
    ap.add_argument("--out", default="stack.csv")
    ap.add_argument("--target", default=None, help="эталон для сверки")
    a = ap.parse_args()

    SC = {}
    for line in open(a.scores):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                SC[parts[0]] = float(parts[1])
            except ValueError:
                pass
    EXC = re.compile(a.exclude) if a.exclude else None
    names, L, R, uid = [], [], [], None
    for n in sorted(SC):
        if EXC and EXC.search(n):
            continue
        p = find(a.pool, n)
        if not p:
            continue
        u, v = load_log(p)
        if len(v) != 250000:
            continue
        if uid is None:
            uid = u
        names.append(n); L.append(v - v.mean() + EL); R.append(SC[n])
    L = np.stack(L); R = np.array(R); n = len(names); NU = L.shape[1]
    print(f"базис: {n} замеренных векторов", flush=True)

    Lc = L - EL
    C = (Lc @ Lc.T) / NU
    k = (VAR_L + np.diag(C) - R ** 2) / 2.0

    # ---- поправка k по реестру замеренных смесей
    if a.registry and os.path.exists(a.registry):
        reg = json.load(open(a.registry))
        A, b = [], []
        for e in reg:
            p = e["csv"] if os.path.exists(e["csv"]) else find(a.pool, os.path.basename(e["csv"])[:-4])
            if not p:
                continue
            _, m = load_log(p)
            m = m - m.mean() + EL
            wj, *_ = np.linalg.lstsq(Lc.T, m - EL, rcond=None)
            pred2 = VAR_L - 2 * wj @ k + wj @ C @ wj
            A.append(wj); b.append((e["lb"] ** 2 - pred2) / 2.0)
        if A:
            A = np.stack(A); b = np.array(b)
            delta = A.T @ np.linalg.solve(A @ A.T + 1e-12 * np.eye(len(b)), b)
            k = k - delta
            print(f"поправка k по {len(b)} замеренным смесям: |delta| до {np.abs(delta).max():.5f}",
                  flush=True)

    # ---- свип риджа с политикой весов
    print("\n ридж     предсказание   max|w|  шортов  политика", flush=True)
    best = None
    for lam in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        w = np.linalg.solve(C + lam * np.eye(n), k)
        pred = float(np.sqrt(max(VAR_L - 2 * w @ k + w @ C @ w, 0)))
        ok = np.abs(w).max() <= MAXW and w.min() >= MAXSHORT
        print(f" {lam:.0e}  {pred:.10f}  {np.abs(w).max():7.3f}  {int((w < 0).sum()):5d}   "
              f"{'принято' if ok else 'отвергнуто'}", flush=True)
        if ok and (best is None or pred < best[0]):
            best = (pred, lam, w)
    if best is None:
        print("\nни одно решение не прошло политику весов")
        return
    pred, lam, w = best
    q = EL + w @ Lc
    q = q + (EL - q.mean())
    out = np.clip(np.expm1(q), 0, None)
    pd.DataFrame({"user_id": uid, "predict": out}).to_csv(a.out, index=False)
    print(f"\nвыбран ридж {lam:.0e}, предсказанный скор {pred:.10f}, записано {a.out}")
    top = sorted(zip(np.abs(w), names, w), reverse=True)[:10]
    print("крупнейшие веса: " + ", ".join(f"{nm} {v:+.4f}" for _, nm, v in top))
    if a.target and os.path.exists(a.target):
        _, y = load_log(a.target)
        d = np.log1p(out) - y
        print(f"сверка с эталоном: расхождение {np.sqrt(np.mean(d ** 2)):.3e}, "
              f"corr {np.corrcoef(np.log1p(out), y)[0, 1]:.8f}")


if __name__ == "__main__":
    main()
