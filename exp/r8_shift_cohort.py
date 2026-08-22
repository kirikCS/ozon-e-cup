"""Сдвинутые окна и когортные агрегаты — третья и четвёртая семьи из разбора Favorita.

r4 взял две семьи из четырёх и дал -6,01e-04 на стенде (около -7,7e-05 на стеке после усадки
в 7,8 раза). Остались две:

  * СДВИНУТЫЕ КОПИИ ОКОН (у победителя суффикс `_2`, сдвиг -7 дней): тот же веер, посчитанный
    от более ранней точки. Даёт модели производную по всем окнам сразу, без явных разностей.
  * АГРЕГАТЫ ВЕРХНЕГО УРОВНЯ тем же кодом, приджойненные обратно: у победителя это item-level
    и (class, store)-level, утроившие число фич почти бесплатно. У нас естественный уровень —
    когорта по активности, плюс отношение клиента к своей когорте.

Глобальный дневной итог по всем 250k внутри якоря константен и межклиентного сигнала не несёт,
но МЕЖДУ якорями меняется — он говорит модели, в каком режиме снят этот якорь. Поэтому включён
отдельно, а не как отношение.

Замер тот же: бустинг на тех же якорях, разница только в колонках, плюс допуск в стенд.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np
import lightgbm as lgb
from pathlib import Path
from common import panel, target, rmsle, build_features, NU, OUT

SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
CH = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
      "scart", "sord", "sday", "catday"]
HALF, WINS, SHIFTS = [7, 21, 60, 180], [7, 30, 90, 365], [0, 7, 28]
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
print("панели ...", flush=True)
P = {c: panel(c).astype(np.float32) for c in CH}

def core(anchor):
    """затухания и структура нулей — тот же блок, что в r4"""
    out, names = [], []
    for c in CH:
        M = P[c]
        for T in HALF:
            L = min(int(6 * T), anchor + 1)
            w = np.exp(-np.arange(L) * np.log(2) / T).astype(np.float32)
            seg = M[:, anchor - L + 1:anchor + 1]
            out.append(np.log1p(np.maximum((seg * w[::-1]).sum(1) / w.sum(), 0)))
            names.append(f"{c}_decay{T}")
        for W in WINS:
            L = min(W, anchor + 1)
            seg = M[:, anchor - L + 1:anchor + 1]
            nz = seg > 0
            cnt = nz.sum(1).astype(np.float32)
            idx = np.arange(L, dtype=np.float32)
            last = np.where(cnt > 0, (L - 1) - (nz * idx).max(1), L)
            first = np.where(cnt > 0, (L - 1) - np.where(nz, idx, L).min(1), L)
            mnz = np.where(cnt > 0, seg.sum(1) / np.maximum(cnt, 1), 0.0)
            out += [cnt / L, last / L, first / L, np.log1p(np.maximum(mnz, 0))]
            names += [f"{c}_nzcnt{W}", f"{c}_nzlast{W}", f"{c}_nzfirst{W}", f"{c}_nzmean{W}"]
    return np.stack(out, 1).astype(np.float32), names

def extended(anchor):
    """сдвинутые копии + когортные агрегаты и отношения + глобальный режим якоря"""
    blocks, names = [], []
    base = None
    for sh in SHIFTS:
        b, nm = core(anchor - sh)
        if sh == 0:
            base = b
        blocks.append(b)
        names += nm if sh == 0 else [f"{n}_m{sh}" for n in nm]
    # когорта по активности: дециль числа ненулевых дней gmv за 365
    L = min(365, anchor + 1)
    act = (P["gmv"][:, anchor - L + 1:anchor + 1] > 0).sum(1)
    q = np.quantile(act, np.linspace(0, 1, 11)); q[0] -= 1; q[-1] += 1
    dec = np.clip(np.digitize(act, q[1:-1]), 0, 9)
    coh = np.zeros_like(base)
    for k in range(10):
        m = dec == k
        if m.sum() > 0:
            coh[m] = base[m].mean(0)
    blocks += [coh, base - coh]
    names += [f"{n}_coh" for n in names[:base.shape[1]]] + \
             [f"{n}_vscoh" for n in names[:base.shape[1]]]
    # режим якоря: глобальные средние по каналам (одинаковы внутри якоря, разные между ними)
    glob = np.tile(base.mean(0), (NU, 1)).astype(np.float32)
    blocks.append(glob[:, :len(CH) * len(HALF)])
    names += [f"{n}_glob" for n in names[:len(CH) * len(HALF)]]
    return np.hstack(blocks).astype(np.float32), names

print("строю базовый фичестор ...", flush=True)
X, base_names = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
t0 = time.time(); EXT = {}
for a in sorted(set(SEL_TR + [SEL_EV])):
    EXT[a], ext_names = extended(a)
    print(f"  якорь {a}: +{EXT[a].shape[1]} колонок ({time.time()-t0:.0f}s)", flush=True)
print(f"базовых {len(base_names)}, расширенных {len(ext_names)}", flush=True)

ytr = np.concatenate([np.log1p(target(a)) for a in SEL_TR], 0)
PAR = dict(n_estimators=900, learning_rate=0.04, num_leaves=95, min_child_samples=150,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
           n_jobs=-1, verbose=-1)
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))
VEC = {}
print(f"\n{'рука':>22} {'колонок':>9} {'RMSLE':>10} {'сек':>7}", flush=True)
for nm, get in (("базовый + r4 (385)", lambda a: np.hstack([X[a], EXT[a][:, :220]])),
                ("базовый + всё",      lambda a: np.hstack([X[a], EXT[a]]))):
    t0 = time.time()
    Xtr = np.concatenate([get(a) for a in SEL_TR], 0)
    m = lgb.LGBMRegressor(**PAR); m.fit(Xtr, ytr)
    v = m.predict(get(SEL_EV)).astype(np.float64)
    VEC[nm] = v
    np.save(OUT / f"r8_{'r4' if 'r4' in nm else 'all'}.npy", v.astype(np.float32))
    print(f"{nm:>22} {Xtr.shape[1]:9d} {sc(v):10.5f} {time.time()-t0:7.0f}", flush=True)
    del Xtr, m

BASE = {}
for p in sorted(Path(OUT).glob("*.npy")):
    if p.stem.startswith(("r1_","r2_","r4_","r5_","r6_","r7_","r8_","p8_","p9_")): continue
    try: v = np.load(p)
    except Exception: continue
    if v.shape != (NU,) or not np.isfinite(v).all(): continue
    if sc(v.astype(np.float64)) < 1.80: BASE[p.stem] = v.astype(np.float64) - v.mean()
print(f"\nстенд: {len(BASE)} векторов")
def fit_eval(cols, fm, em, lam=1e-4):
    M = np.stack(cols, 1)
    A = np.concatenate([M[fm], np.ones((fm.sum(),1))], 1)
    G = A.T@A + lam*len(A)*np.eye(A.shape[1]); G[-1,-1] -= lam*len(A)
    w = np.linalg.solve(G, A.T@ly[fm])
    v = M@w[:-1] + w[-1]
    return rmsle(y[em], np.expm1(np.clip(v[em],0,13)))
cols = list(BASE.values())
rng = np.random.default_rng(0); half = rng.random(NU) < 0.5
SPL = ((half,~half),(~half,half))
b = [fit_eval(cols,f,e) for f,e in SPL]
print(f"стенд без кандидатов: {np.mean(b):.6f}")
rs = np.random.default_rng(11)
print(f"\n{'кандидат':>22} {'A→B':>11} {'B→A':>11} {'среднее':>11} {'контроль':>11} {'на стеке':>11}")
for nm, v in VEC.items():
    vv = v - v.mean(); vp = rs.permutation(vv)
    d  = [fit_eval(cols+[vv],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    dc = [fit_eval(cols+[vp],f,e)-b[i] for i,(f,e) in enumerate(SPL)]
    print(f"{nm:>22} {d[0]:+11.2e} {d[1]:+11.2e} {np.mean(d):+11.2e} {np.mean(dc):+11.2e} "
          f"{np.mean(d)/7.8:+11.2e}")
print("\nпоследняя колонка — ожидание на настоящем стеке (усадка 7,8, измерена на q99)")
print("\ndone")
