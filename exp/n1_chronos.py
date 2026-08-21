"""Chronos-2 в трёх режимах, включая гибрид с нашим хазардом.

Chronos-2: 120М, encoder-only в духе T5, контекст до 8192, выход КВАНТИЛЬНЫЙ, поддерживает
past_covariates и — что для нас новое — future_covariates. У Toto последнего не было, а целевое
окно содержит 23 февраля и 8 марта, даты фиксированные.

Но урок уже извлечён двумя независимыми семействами. Toto (квантили): P(Y>0) = 0.9967 против
эмпирических 0.5631. Гауссова диффузия, обученная НА НАШИХ данных: P(Y>0) = 1.0000 ровно. Ни одна
непрерывная плотность не воспроизводит атом в нуле, сколько её ни обучай.

Поэтому zero-shot здесь идёт только как контроль, а основная рука — ГИБРИД.

Как из безусловных квантилей достать условное распределение. Если истинная доля положительных p,
то условная функция распределения при y>0 это F_cond(x) = (F(x) − (1−p)) / p. Значит выборка из
условной части — это u_cond = (1−p) + p·u при u ~ U(0,1), пропущенное через обратную функцию
Chronos. Дальше гейт: с вероятностью p берём это значение, иначе ноль.

Смысл: Chronos отвечает за ФОРМУ положительной части, наш хазард — за НУЛЕВУЮ МАССУ. Каждый
делает то, что умеет.

  zs        сырые квантили + гауссова копула — контроль, повторение Toto
  hybrid    условная часть от Chronos × наш обученный хазард
  covar     то же плюс known future covariates: дни недели и флаги праздников горизонта
"""
import sys, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from datetime import date, timedelta
from scipy.stats import norm

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from chronos import Chronos2Pipeline

DEV = "cuda"
A = 348
CTX = int(os.environ.get("CTX", 320))
NSUB = int(os.environ.get("NSUB", 20000))
NPATH = int(os.environ.get("NPATH", 256))
BS = int(os.environ.get("BS", 256))
QL = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
D0 = date(2025, 1, 1)
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)

rng = np.random.default_rng(0)
sub = np.sort(rng.choice(NU, NSUB, replace=False))
y = target(A)[sub]; ly = np.log1p(y)
emp = float((y > 0).mean())
print(f"подвыборка {NSUB}, путей {NPATH}, контекст {CTX}", flush=True)
print(f"эмпирическая P(Y>0) = {emp:.4f}", flush=True)

def wnd(M, users, end, n):
    lo = end - n + 1
    if lo >= 0:
        return M[users, lo:end + 1].astype(np.float32)
    got = M[users, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), -lo), np.float32), got], 1)

# ---- наш хазард: свёрточный ствол -> 30 логитов, обучается на наших якорях
CH_N = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32)) if panel(n).max() > 3
                 else panel(n).astype(np.float32)).astype(np.float16) for n in CH_N], 1)
NCH = SEQ.shape[1]
TR = [318, 290, 262, 234, 206]

def sw(users, end, n=182):
    lo = end - n + 1
    if lo >= 0:
        return SEQ[users, :, lo:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), NCH, -lo), np.float32), got], 2)

class Gate(nn.Module):
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.head = nn.Sequential(nn.Linear(2 * h, 128), nn.BatchNorm1d(128), nn.SiLU(),
                                  nn.Linear(128, HOR))
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))

print("обучаю хазард-гейт ...", flush=True)
t0 = time.time()
torch.manual_seed(0); r2 = np.random.default_rng(0)
gate = Gate(NCH).to(DEV)
opt = torch.optim.AdamW(gate.parameters(), lr=1.5e-3, weight_decay=1e-5)
EP, GB = 5, 1024
steps = EP * len(TR) * (NU // GB)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=max(steps, 10))
for ep in range(EP):
    gate.train()
    for a in TR:
        perm = r2.permutation(NU)
        for i in range(0, NU, GB):
            idx = np.sort(perm[i:i + GB])
            if len(idx) < 64: continue
            x = torch.from_numpy(sw(idx, a)).to(DEV)
            pos = torch.from_numpy((G[idx, a + 1:a + 1 + HOR] > 0).astype(np.float32)).to(DEV)
            opt.zero_grad(set_to_none=True)
            Fn.binary_cross_entropy_with_logits(gate(x), pos).backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 5.0); opt.step()
            if sch.last_epoch < steps - 1: sch.step()
gate.eval()
with torch.no_grad():
    PH = np.concatenate([torch.sigmoid(gate(torch.from_numpy(sw(sub[i:i+2048], A)).to(DEV)))
                         .cpu().numpy() for i in range(0, NSUB, 2048)], 0)
print(f"  гейт готов ({time.time()-t0:.0f}s), средняя P(y_d>0) = {PH.mean():.4f}", flush=True)
del gate; torch.cuda.empty_cache()

print("загружаю chronos-2 ...", flush=True)
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV)

def cal_past(users, a, n=CTX):
    """те же календарные признаки на историческом окне — требование Chronos к ковариатам"""
    out = np.zeros((len(users), n, 3), np.float32)
    for i in range(n):
        d = D0 + timedelta(days=a - n + 1 + i)
        out[:, i, 0] = d.weekday() / 6.0
        out[:, i, 1] = 1.0 if d.weekday() >= 5 else 0.0
        out[:, i, 2] = 1.0 if (d.month, d.day) in ((2, 23), (3, 8), (2, 14), (3, 7)) else 0.0
    return out

def cal_future(a, n=HOR):
    out = np.zeros((n, 3), np.float32)
    for i in range(n):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, 0] = d.weekday() / 6.0
        out[i, 1] = 1.0 if d.weekday() >= 5 else 0.0
        out[i, 2] = 1.0 if (d.month, d.day) in ((2, 23), (3, 8), (2, 14), (3, 7)) else 0.0
    return out

def get_q(use_cov):
    out = np.empty((NSUB, len(QL), HOR), np.float32)
    t0 = time.time()
    for i in range(0, NSUB, BS):
        u = sub[i:i + BS]
        tgt = wnd(G, u, A, CTX)
        if use_cov:
            # ключи future_covariates обязаны быть ПОДМНОЖЕСТВОМ past_covariates: модель должна
            # видеть историю ковариаты, чтобы выучить связь. Календарь известен на всю ось,
            # поэтому подаётся и в прошлое, и в будущее; визиты и корзины — только в прошлое.
            cp, cf = cal_past(u, A), cal_future(A)
            vp, cpc = wnd(V, u, A, CTX), wnd(C, u, A, CTX)
            inp = [{"target": tgt[k],
                    "past_covariates": {"dow": cp[k, :, 0], "wknd": cp[k, :, 1],
                                        "hol": cp[k, :, 2], "vis": vp[k], "cart": cpc[k]},
                    "future_covariates": {"dow": cf[:, 0], "wknd": cf[:, 1], "hol": cf[:, 2]}}
                   for k in range(len(u))]
        else:
            inp = [tgt[k] for k in range(len(u))]
        q = pipe.predict_quantiles(inp, prediction_length=HOR, quantile_levels=QL)
        q = q[0] if isinstance(q, tuple) else q          # (список тензоров, что-то ещё)
        # каждый элемент: (переменная, горизонт, квантиль) -> нам нужно (квантиль, горизонт)
        arr = torch.stack([e[0] for e in q], 0).float().cpu().numpy()   # (b, HOR, nq)
        out[i:i + len(u)] = np.transpose(arr, (0, 2, 1))
        if i == 0:
            print(f"    первый батч {time.time()-t0:.1f}s -> на 250k ~"
                  f"{(time.time()-t0)*250000/len(u)/3600:.2f} ч", flush=True)
    return out

def invq(Q, u):
    """обратная функция по квантилям, линейная интерполяция, хвосты обрезаны"""
    ql = np.array(QL)
    idx = np.clip(np.searchsorted(ql, u), 1, len(ql) - 1)
    lo, hi = ql[idx - 1], ql[idx]
    w = (u - lo) / (hi - lo)
    Qs = np.sort(Q, axis=1)
    take = lambda j: np.take_along_axis(Qs, j[:, None, :], 1)[:, 0]
    return take(idx - 1) * (1 - w) + take(idx) * w

def simulate(Q, mode, rho=0.9, seed=0):
    r = np.random.default_rng(seed)
    tot = np.zeros((NSUB,), np.float64); cnt = np.zeros((NSUB,), np.float64)
    acc = np.zeros(NSUB); pos_acc = np.zeros(NSUB)
    for _ in range(NPATH):
        z = r.standard_normal((NSUB, 1)); e = r.standard_normal((NSUB, HOR))
        u = norm.cdf(rho * z + math.sqrt(max(1 - rho ** 2, 0)) * e)
        if mode == "hybrid":
            uc = (1 - PH) + PH * u                      # условная часть при y>0
            val = invq(Q, np.clip(uc, 1e-4, 1 - 1e-4))
            val = np.maximum(val, 0) * (r.random((NSUB, HOR)) < PH)
        else:
            val = np.maximum(invq(Q, np.clip(u, 1e-4, 1 - 1e-4)), 0)
        s = val.sum(1)
        acc += np.log1p(s); pos_acc += (s > 0)
    return acc / NPATH, pos_acc / NPATH

ref = OUT / "h8_gauss.npy"
if ref.exists():
    r_ = np.load(ref)[sub]; r_ = r_ - r_.mean() + ly.mean()
    print(f"\nнаша дневная модель на подвыборке: "
          f"{rmsle(y, np.expm1(np.clip(r_,0,13))):.5f}", flush=True)

print(f"\n{'рука':>10} {'RMSLE':>9} {'P(Y>0)':>9} {'сек':>7}", flush=True)
RES = {}
for tag, cov, mode in (("zs", False, "raw"), ("hybrid", False, "hybrid"), ("covar", True, "hybrid")):
    t0 = time.time()
    try:
        Q = get_q(cov)
        v, pr = simulate(Q, mode)
    except Exception as e:
        import traceback; traceback.print_exc(limit=2)
        print(f"{tag:>10}  ОШИБКА {type(e).__name__}: {str(e)[:110]}", flush=True)
        continue
    RES[tag] = v
    sh = v - v.mean() + ly.mean()
    print(f"{tag:>10} {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} {pr.mean():9.4f} "
          f"{time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"n1_chronos_{tag}.npy", v)
if ref.exists():
    for k, v in RES.items():
        print(f"corr({k}, наша дневная) = {np.corrcoef(v, np.load(ref)[sub])[0,1]:.5f}")
print("\nчитать так: если hybrid резко лучше zs — подтверждено, что дело в атоме нуля,")
print("и любую непрерывную модель надо сочетать с явным гейтом")
print("\ndone")
