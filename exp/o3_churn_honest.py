"""Честная проверка поправки на отток: подгонка и оценка на разных клиентах и двух протоколах.

В o2 свипы дали мелкие выигрыши (2,1e-5 у жёсткого порога, 5,9e-5 у мягкой поправки), но параметр
подбирался на том же якоре, где мерился, а опора была одиночной свёрткой без масштабной
калибровки. Дециль-таблица показала пере-разброс: у надёжных клиентов остаток -0,047, у ушедших
+0,026. Часть этого снимает обычная аффинная калибровка, которую стек делает и так.

Здесь: сначала аффинная калибровка (это база, а не улучшение), поверх неё — поправки на отток,
все параметры подгоняются на ПОЛОВИНЕ клиентов и оцениваются на другой, обе стороны. Это ровно
тот же вопрос, что публика против привата: переносится ли поправка на клиентов, которых не видела.
Плюс второй протокол HON (якорь 408) со своим классификатором.
"""
import sys
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from common import target, rmsle, build_features, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], 408

def churn_model(TR, EV):
    X, _ = build_features(sorted(set(TR + [EV])), verbose=False)
    Xtr = np.concatenate([X[a] for a in TR], 0)
    ytr = np.concatenate([(target(a) > 0).astype(np.int8) for a in TR], 0)
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)
    return 1.0 - clf.predict_proba(X[EV])[:, 1]

def affine(v, ly, m):
    """a*v + b по МНК на маске m — это база, стек делает так же"""
    A = np.stack([v[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, ly[m], rcond=None)[0]
    return a * v + b

def sc(v, y, m):
    return rmsle(y[m], np.expm1(np.clip(v[m], 0, 13)))

PROTO = []
for tag, TR, EV, cands in (("SEL", SEL_TR, SEL_EV, ["h8_gauss"]),
                           ("HON", FIN_TR, FIN_EV, ["h6_hon_all", "h5_hon_base"])):
    base = None
    for c in cands:
        p = OUT / f"{c}.npy"
        if p.exists():
            v = np.load(p)
            y_ = target(EV)
            if len(v) == len(y_):
                s = rmsle(y_, np.expm1(np.clip(v - v.mean() + np.log1p(y_).mean(), 0, 13)))
                if s < 1.85:                     # осмысленный вектор именно этого якоря
                    base = (c, v); break
    if base is None:
        print(f"[{tag}] опорного вектора нет, пропускаю", flush=True); continue
    PROTO.append((tag, TR, EV, base))
    print(f"[{tag}] опора {base[0]}, якорь {EV}", flush=True)

rng = np.random.default_rng(0)
for tag, TR, EV, (bname, v0) in PROTO:
    y = target(EV); ly = np.log1p(y)
    print(f"\n{'='*66}\n[{tag}] опора «{bname}», якорь {EV}", flush=True)
    pch = churn_model(TR, EV)
    from sklearn.metrics import roc_auc_score
    print(f"классификатор: AUC {roc_auc_score((y>0).astype(int), 1-pch):.5f}   "
          f"средняя p_отток {pch.mean():.4f} против факта {(y==0).mean():.4f}", flush=True)

    half = rng.random(len(y)) < 0.5
    print(f"\n{'поправка':>24} {'подгонка A→B':>14} {'подгонка B→A':>14} {'среднее':>11}")
    def both(make):
        d = []
        for fit_m, ev_m in ((half, ~half), (~half, half)):
            vb = affine(v0, ly, fit_m)
            s0 = sc(vb, y, ev_m)
            vv = make(vb, fit_m)
            d.append(sc(vv, y, ev_m) - s0)
        return d
    print(f"{'аффинная база':>24} {'—':>14} {'—':>14} {'0':>11}")
    for tname, mk in (
        ("жёсткий ноль, порог по A", lambda vb, fm: hard(vb, fm)),
        ("мягкая (1-p)^alpha по A",  lambda vb, fm: soft(vb, fm)),
        ("подецильные сдвиги по A",  lambda vb, fm: dec(vb, fm)),
    ):
        def hard(vb, fm):
            best, bt = 1e9, None
            for t in (0.99, 0.98, 0.97, 0.95, 0.92, 0.90):
                w = vb.copy(); w[pch > t] = 0.0
                s = sc(w, y, fm)
                if s < best: best, bt = s, t
            w = vb.copy(); w[pch > bt] = 0.0; return w
        def soft(vb, fm):
            best, ba = 1e9, 0.0
            for al in np.arange(-0.15, 0.31, 0.01):
                w = vb * (1 - pch) ** al
                s = sc(w, y, fm)
                if s < best: best, ba = s, al
            return vb * (1 - pch) ** ba
        def dec(vb, fm):
            q = np.quantile(pch, np.linspace(0, 1, 11)); q[0] -= 1e-9; q[-1] += 1e-9
            d = np.clip(np.digitize(pch, q[1:-1]), 0, 9)
            w = vb.copy()
            for k in range(10):
                m = d == k
                w[m] += (ly[m & fm] - vb[m & fm]).mean()
            return w
        dd = both(mk)
        print(f"{tname:>24} {dd[0]:+14.2e} {dd[1]:+14.2e} {np.mean(dd):+11.2e}")
print("\nотрицательное = поправка помогла на невиданных клиентах\ndone")
