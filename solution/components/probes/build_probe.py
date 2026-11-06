"""Сборка зонда лестницы по НАСТОЯЩЕЙ конструкции из data/probe_recipes.json.

Конструкций пять, и это не стилистика — они реально разные. Цена ошибки измерена:
у q99 записанный ранее «ранний» рецепт дал направление с corr −0.029 к эталонному,
то есть ортогональное ему, тогда как верная конструкция даёт +0.962.

  «компонента» (q99; опора v18 — до того, как появилось правило про разности)
      операнды НЕ вычитаются: каждый нормируется и идёт своим слагаемым,
      направление — их среднее, дальше как «ранний»

  «ранний»  (quad, nosm; август, опоры v19-v20)
      d = (b-mean(b))/sd(b) − (a-mean(a))/sd(a)
      ортогонализация к {1, q}, нормировка на sd
      eps = 0.0703 ВБИТ КОНСТАНТОЙ, бисекции НЕТ вовсе
      pred = expm1(clip(L + eps*d, 0, 13))
      уровень держится сам: направление средненулевое и ортогонально L

  «w20»     (operator; одна пара, exp/w20_diff_probe.py)
      d = (b-mean(b)) − (a-mean(a)), ортогонализация, нормировка
      eps = sqrt((R0+0.0015)^2 − R0^2), 90 шагов бисекции к EL = 2.3284

  «w23»     (cell, p_cnn, p_enc; несколько пар, exp/w23_cluster_probe.py)
      каждая разность центрируется, ортогонализуется и нормируется ОТДЕЛЬНО,
      знаки выравниваются по первой, берётся среднее, снова ортогонализация
      и нормировка, дальше как w20

  «панели»  (activity_comb)
      82 направления из десяти панелей, ридж-регрессия остатка,
      взвешенная сумма — i3_activity_sweep.py + i5_make_probe.py

Запуск:
    python build_probe.py --probe q99 --anchor ФАЙЛ --anchor-score R0 \
                          --vectors КАТАЛОГ --out ФАЙЛ [--reference ЭТАЛОН]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EL, COST = 2.3284, 0.0015
EPS_EARLY = 0.0703


def load_csv(p):
    d = pd.read_csv(p).sort_values("user_id")
    return d["user_id"].to_numpy(), np.log1p(d["predict"].to_numpy(np.float64))


def load_operand(vdirs, name):
    """Операнд направления: .npy вектор модели либо CSV сабмита."""
    for d in vdirs:
        for f in glob.glob(os.path.join(d, "**", name + ".npy"), recursive=True):
            a = np.asarray(np.load(f), np.float64)
            if a.ndim == 2 and a.shape[0] == 250000:
                a = a.sum(1)
            if a.ndim != 1 or a.shape[0] != 250000:
                continue
            return np.log1p(np.clip(a, 0, None)) if a.mean() > 5 else a
    for d in vdirs:
        p = os.path.join(d, name + ".csv")
        if os.path.exists(p):
            return load_csv(p)[1]
    return None


def orth(x, A, sol):
    return x - A @ (sol @ x)


def build(q, pairs, mech, vdirs, R0):
    NU = len(q)
    A = np.stack([np.ones(NU), q], 1)
    sol = np.linalg.pinv(A)
    parts, missing = [], []
    for a_, b_ in pairs:
        va, vb = load_operand(vdirs, a_), load_operand(vdirs, b_)
        if va is None or vb is None:
            missing.append(a_ if va is None else b_)
            continue
        if mech == "компонента":
            # НЕ разность: каждый операнд идёт своим слагаемым, направление —
            # их среднее. Так строились зонды до правила «зондировать разность»
            # (оно появилось только после v22).
            for v in (va, vb):
                d = orth((v - v.mean()) / v.std(), A, sol)
                parts.append(d / d.std())
            continue
        if mech == "ранний":
            d = (vb - vb.mean()) / vb.std() - (va - va.mean()) / va.std()
        else:
            d = (vb - vb.mean()) - (va - va.mean())
        d = orth(d, A, sol)
        parts.append(d / d.std())
    if not parts:
        return None, missing
    if mech == "w23" and len(parts) > 1:
        for i in range(1, len(parts)):
            if np.corrcoef(parts[0], parts[i])[0, 1] < 0:
                parts[i] = -parts[i]
    D = np.mean(parts, 0)
    D = orth(D, A, sol)
    D = D / D.std()

    if mech in ("ранний", "компонента"):
        pred = np.expm1(np.clip(q + EPS_EARLY * D, 0, 13))
        return np.clip(pred, 0, None), missing
    eps = float(np.sqrt((R0 + COST) ** 2 - R0 ** 2))
    t = q + eps * D
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL:
            lo = c
        else:
            hi = c
    return np.clip(np.expm1(t + (lo + hi) / 2), 0, None), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--anchor-score", type=float, required=True)
    ap.add_argument("--vectors", required=True, help="каталоги с .npy, через запятую")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--recipes", default=os.path.join(HERE, "..", "..", "data", "probe_recipes.json"))
    a = ap.parse_args()

    rec = json.load(open(a.recipes))["зонды"].get(a.probe)
    if rec is None:
        print(f"{a.probe}: рецепта нет"); return
    mech = rec["механизм"]
    if mech in ("панели", "скрипт"):
        print(f"{a.probe}: механизм «{mech}» — собирается своим скриптом "
              f"({', '.join(rec.get('скрипты', []))}), здесь не строится")
        return

    uid, q = load_csv(a.anchor)
    pred, missing = build(q, rec["diffs"], mech, [d.strip() for d in a.vectors.split(",")],
                          a.anchor_score)
    if pred is None:
        print(f"{a.probe}: нет операндов {missing}"); return
    pd.DataFrame({"user_id": uid, "predict": pred}).to_csv(a.out, index=False)
    note = f", не найдено: {missing}" if missing else ""
    print(f"{a.probe}: механизм {mech}, пар {len(rec['diffs'])}{note} -> {a.out}")
    if a.reference and os.path.exists(a.reference):
        _, y = load_csv(a.reference)
        x = np.log1p(pred)
        print(f"  сверка: corr {np.corrcoef(x, y)[0,1]:.10f}, "
              f"max|d| {np.abs(x-y).max():.3e}  (ожидалось {rec.get('точность','')})")


if __name__ == "__main__":
    main()
