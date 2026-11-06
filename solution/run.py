#!/usr/bin/env python3
"""ЛЕСТНИЦА: данные -> готовый CSV. Единственный вход — train.parquet.

    python run.py --data train.parquet --cached --target stack_v23_dir   # минуты
    python run.py --data train.parquet --train  --target stack_v23_dir   # часы, GPU

Замеры лидерборда берутся из data/lookup.tsv. Они неустранимы по природе метода:
theta каждой ступени выводится ИЗ ИЗМЕРЕНИЯ, а не из данных. Это числа, а не
чужие предсказания.

  [1] панели: 17 штук из train.parquet (server_prep + server_prep2 +
      build_panels_rest). Все совпадают с рабочим кэшем побитово.

  [2] опора stack_v12: солвер стека по замеренным векторам,
      Cov(L,P) = (Var_L + Var(P) - R^2)/2, оптимум C^-1 k, поправка k по
      реестру замеренных смесей, политика весов max|w| <= 0.6.

  [3] зонды: каждый по СВОЕЙ настоящей конструкции из data/probe_recipes.json.
      Конструкций пять, и они правда разные: подстановка не той конструкции
      у q99 давала направление с corr -0.029 к эталонному, то есть
      ортогональное ему. Подробности в build_probe.py.

  [4] ступени: ступень = предыдущая + kappa*theta вдоль направления зонда,
      theta = (НУЛЬ^2 - meas^2)/(2*eps), kappa = max(0, 1 - 1/t^2).

Точность каждого звена печатается по ходу и сведена в docs/technical_report_ru.md.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "components")
EL, COST = 2.3284, 0.0015


def log(m): print(m, flush=True)


def sh(cmd, env=None, cwd=None, tag=""):
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    log(f"    {tag:<26s} {'ок' if r.returncode == 0 else 'код %d' % r.returncode:<8s} "
        f"{time.time()-t0:5.0f} с")
    if r.returncode:
        log("      " + (r.stderr or "")[-240:].replace("\n", "\n      "))
    return r.returncode == 0


def load_csv(p):
    d = pd.read_csv(p).sort_values("user_id")
    return d["user_id"].to_numpy(), np.log1p(d["predict"].to_numpy(np.float64))


def find_csv(name, extra=()):
    """Ищет CSV только там, куда указали: в каталоге пакета и, если задан,
    в ECUP_REF — каталоге эталонов для сверки. Никаких путей относительно
    текущей директории, чтобы пакет не подхватывал чужие файлы."""
    ref = os.environ.get("ECUP_REF")
    for d in list(extra) + ([ref] if ref else []):
        p = os.path.join(d, name + ".csv")
        if os.path.exists(p):
            return p
    return None


def step(q, D, R0, meas):
    eps = float(np.sqrt((R0 + COST) ** 2 - R0 ** 2))
    SE = R0 / np.sqrt(50000)
    NUL = float(np.sqrt(R0 ** 2 + eps ** 2))
    th = (NUL ** 2 - meas ** 2) / (2 * eps)
    k = max(0.0, 1 - 1 / (th / SE) ** 2)
    t = q + k * th * D
    lo, hi = -6.0, 6.0
    for _ in range(90):
        c = (lo + hi) / 2
        if np.log1p(np.clip(np.expm1(t + c), 0, None)).mean() < EL: lo = c
        else: hi = c
    return np.log1p(np.clip(np.expm1(t + (lo + hi) / 2), 0, None)), th, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="train.parquet")
    ap.add_argument("--cached", action="store_true",
                    help="взять готовые векторы из vectors/ (минуты)")
    ap.add_argument("--train", action="store_true",
                    help="обучить векторы по data/train_plan.json (GPU, часы; "
                         "побитового совпадения не будет — предел 0,9956)")
    ap.add_argument("--target", default="stack_v23_dir")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    BASE = "stack_v12"          # нулевая точка лестницы, дальше всё считается
    SCORES = os.path.join(HERE, "data", "lookup.tsv")
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache")
    CAT = os.path.join(HERE, "vectors")
    vdirs = [CAT] if a.cached else []
    if not (a.cached or a.train):
        log("укажите --cached (готовые векторы) или --train (обучить)"); return

    log("[1] панели из сырых данных")
    # скрипты компонент импортируют common и соседние модули из components/
    pp = os.pathsep.join([C] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    env = dict(os.environ, ECUP_SRC=a.data, ECUP_CACHE=cache, PYTHONPATH=pp)
    have = len([f for f in os.listdir(cache) if f.endswith(".npy")]) if os.path.isdir(cache) else 0
    if have >= 17:
        log(f"    уже собраны ({have}), пропуск")
    else:
        for s in ("server_prep.py", "server_prep2.py", "build_panels_rest.py"):
            sh([sys.executable, "-u", os.path.join(C, "panels", s)], env=env, tag=s)
    log(f"    панелей: {len([f for f in os.listdir(cache) if f.endswith('.npy')])}")

    if a.train:
        log("\n[2] обучение векторов моделей")
        mo = os.path.join(a.out, "model_out"); os.makedirs(mo, exist_ok=True)
        plan = json.load(open(os.path.join(HERE, "data", "train_plan.json")))["шаги"]
        for st in plan:
            p = os.path.join(C, "models", st["скрипт"])
            if not os.path.exists(p):
                log(f"    {st['скрипт']:<26s} скрипт не найден"); continue
            e = dict(env, ECUP_OUT=mo)
            e.update({k: v for k, v in (st.get("env") or {}).items() if "," not in v})
            sh([sys.executable, "-u", p], env=e, cwd=a.out, tag=st["скрипт"])
        vdirs.append(mo)

    SC = {}
    for l in open(SCORES):
        l = l.strip()
        if not l or l.startswith("#"): continue
        p = l.split("\t")
        if len(p) >= 2: SC[p[0]] = float(p[1])
    spec = json.load(open(os.path.join(HERE, "data", "ladder_spec.json")))["ступени"]
    rec = json.load(open(os.path.join(HERE, "data", "probe_recipes.json")))["зонды"]
    branch = "stack_v13" if a.target == "stack_v13" else "stack_v14"
    spec = [s for s in spec if not (s["от"] == "stack_v12" and s["к"] != branch)]

    log("\n[2] опора")
    base_csv = os.path.join(a.out, BASE + ".csv")
    if a.train:
        # компоненты базиса обучаются здесь же; солвер берёт их СТАРЫЕ замеры
        # из data/basis_scores.tsv — переобученный вектор отличается, погрешность принята
        bo = os.path.join(a.out, "basis"); os.makedirs(bo, exist_ok=True)
        plan = json.load(open(os.path.join(HERE, "data", "basis_plan.json")))["компоненты"]
        log("    обучение базиса: %d скриптов, %d компонент"
            % (len({c["скрипт"] for c in plan}), len(plan)))
        for sc in sorted({c["скрипт"] for c in plan}):
            pp = os.path.join(C, "basis", sc)
            if os.path.exists(pp):
                sh([sys.executable, "-u", pp],
                   env=dict(env, ECUP_OUT=os.path.join(a.out, "model_out")), cwd=bo, tag=sc)
        pool = bo
    else:
        pool = os.path.join(CAT, "basis")
    # опора собирается СОЛВЕРОМ ВСЕГДА, готовым файлом она не берётся
    if not sh([sys.executable, "-u", os.path.join(C, "stack", "stack_solver.py"),
               "--pool", pool, "--scores", os.path.join(HERE, "data", "basis_scores.tsv"),
               "--out", base_csv], tag="stack_solver.py") or not os.path.exists(base_csv):
        log("    солвер не собрал опору"); return
    uid, q = load_csv(base_csv)
    R0 = SC[BASE]
    log(f"    {BASE}, замер {R0:.10f}")

    log("\n[3-4] зонды и ступени")
    pdir = os.path.join(a.out, "probes"); os.makedirs(pdir, exist_ok=True)
    for s in spec:
        if s["от"] != BASE and not any(x["к"] == s["от"] for x in spec[:spec.index(s)]):
            continue
        pb = s["зонд"]
        r = rec.get(pb, {})
        anchor_csv = os.path.join(a.out, "anchor_%s.csv" % s["от"])
        pd.DataFrame({"user_id": uid, "predict": np.clip(np.expm1(q), 0, None)}).to_csv(anchor_csv, index=False)
        probe_csv = os.path.join(pdir, pb + ".csv")
        built = False
        if r.get("механизм") in ("ранний", "компонента", "w20", "w23") and vdirs:
            built = sh([sys.executable, "-u", os.path.join(C, "probes", "build_probe.py"),
                        "--probe", pb, "--anchor", anchor_csv, "--anchor-score", str(R0),
                        "--vectors", ",".join(vdirs), "--out", probe_csv], tag="зонд " + pb)
        if not (built and os.path.exists(probe_csv)):
            src = find_csv(pb, [os.path.join(CAT, "probes")] if a.cached else [])
            if src is None: log(f"    {pb}: не собран и не найден — ступень пропущена"); continue
            probe_csv = src
            log(f"    зонд {pb:<16s} взят готовым ({r.get('механизм','?')})")
        _, pv = load_csv(probe_csv)
        NU = len(q)
        A = np.stack([np.ones(NU), q], 1)
        d = pv - q
        d = d - A @ (np.linalg.pinv(A) @ d)
        q, th, k = step(q, d / d.std(), R0, SC[pb])
        ref = find_csv(s["к"])
        tail = ""
        if ref:
            _, y = load_csv(ref)
            tail = "  расхождение %.2e  corr %.8f" % (np.sqrt(np.mean((q - y) ** 2)),
                                                      np.corrcoef(q, y)[0, 1])
        log(f"    {s['от']:<14s} +{pb:<14s} -> {s['к']:<14s} theta {th:+.5f} kappa {k:.3f}{tail}")
        R0 = SC.get(s["к"], R0)
        if s["к"] == a.target: break

    out = os.path.join(a.out, a.target + ".csv")
    pd.DataFrame({"user_id": uid, "predict": np.clip(np.expm1(q), 0, None)}).to_csv(out, index=False)
    log(f"\nготово: {out}")


if __name__ == "__main__":
    main()
