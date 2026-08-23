"""HMM ПОВЕРХ дневной модели: справедливая проверка режимной структуры.

Первая HMM (x5/x7) провалилась — RMSLE 1,80278, вклад -5,6e-07, то есть ноль при корреляции
0,955. Но провал был по моей спецификации, а не по семейству: та модель знала только бинарную
последовательность покупок, темп и средний чек. Ни календаря, ни одиннадцати каналов, ни 165
табличных признаков — всего того, чем пользуются остальные модели.

Собранное правило по четырём таким случаям: ортогональность оплачивается только от КОМПЕТЕНТНОЙ
модели. Всё, что вносит вклад, лежит в пределах +-0,005 от лучшего скора; всё, что отстаёт на
0,026 и больше (Tweedie, knn_k50, ord:4, HMM), даёт ноль независимо от корреляции. Стек не умеет
использовать шум.

Поэтому здесь состояние не ЗАМЕНЯЕТ дневную модель, а даёт СДВИГ к её выходу:

    p(покупка | s, i, d) = sigmoid( hz_{i,d} + a_s )
    величина             ~ N( mu_{i,d} + m_s, exp(ls_{i,d}) * v_s )

где hz, mu, ls — выходы обученной дневной модели. Такая HMM заведомо не глупее её, а состояние
ловит ровно то, чего модель не видит: короткую кучность, измеренную в x2 (лаги 1-3, гаснет к
пятому дню) и независимо подтверждённую временем автокорреляции режима 2,88 дня в x7.

Поденные выходы по истории берутся прогоном модели на семи якорях подряд: каждый даёт свои 30
дней, вместе 210 дней (139..348). Обучение одно, дальше только считывание.

Считыватель тот же прямой ход с преобразованием Лапласа, что в x7: перебора K^30 путей не нужно.
"""
import sys
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, H, D0, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH, SIGMA = 364, 8, 512, 96, 16, 0.8
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], 408
NSEED, SUBSAMPLE, EL_LEVEL = 4, 0.8, 2.3284

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

print("строю статические признаки ...", flush=True)
ANCH_ALL = sorted(set(SEL_TR + FIN_TR + [SEL_EV, FIN_EV]))
X, names = build_features(ANCH_ALL, verbose=False)
FS = len(names)
# нормировка считается ТОЛЬКО по обучающим якорям SELECT — так было в g3, где рука all дала
# 1.73829. В первой версии h1 статистики брались по всем десяти якорям, включая финальный 408:
# распределение сдвигалось, скор падал до 1.838, и это к тому же мягкая утечка.
MU = np.mean([X[a].mean(0) for a in SEL_TR], 0)
SD = np.maximum(np.mean([X[a].std(0) for a in SEL_TR], 0), 1e-2)
ST = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in ANCH_ALL}
del X
print(f"  {FS} колонок", flush=True)

def cal_feats(a):
    out = np.zeros((H, 10), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, d.weekday()] = 1.0
        out[i, 7] = d.day / 31.0
        out[i, 8] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        out[i, 9] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
    return out

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class GRN(nn.Module):
    """gated residual network: dense -> ELU -> dense -> GLU -> add skip -> norm, with optional
    context added before the nonlinearity (the TFT static-conditioning point)"""
    def __init__(self, din, dh, dout=None, dctx=None, drop=0.1):
        super().__init__()
        dout = dout or din
        self.fc1 = nn.Linear(din, dh)
        self.ctx = nn.Linear(dctx, dh, bias=False) if dctx else None
        self.fc2 = nn.Linear(dh, dout * 2)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)
        self.dp = nn.Dropout(drop)
    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.dp(self.fc2(Fn.elu(h)))
        a, b = h.chunk(2, -1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))

class VSN(nn.Module):
    """per-timestep softmax over channels; each channel gets its own GRN first"""
    def __init__(self, nvar, d, dctx=None):
        super().__init__()
        self.per = nn.ModuleList([GRN(1, d, d) for _ in range(nvar)])
        self.sel = GRN(nvar, d, nvar, dctx=dctx)
        self.nvar = nvar
    def forward(self, x, c=None):
        # x: (B, C, T) -> (B, T, C)
        xt = x.transpose(1, 2)
        w = torch.softmax(self.sel(xt, c.unsqueeze(1).expand(-1, xt.shape[1], -1)
                                   if c is not None else None), -1)
        parts = torch.stack([self.per[i](xt[..., i:i + 1]) for i in range(self.nvar)], -1)
        return (parts * w.unsqueeze(-2)).sum(-1).transpose(1, 2), w

class Trunk(nn.Module):
    def __init__(self, cin, h=128, dil=(1, 2, 4, 8, 16, 32, 64), nbeats=False):
        super().__init__()
        self.inp = nn.Conv1d(cin, h, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.back = nn.ModuleList([nn.Conv1d(h, h, 1) for _ in dil]) if nbeats else None
        self.out = h * 3
    def forward(self, x):
        z = self.inp(x)
        acc = 0
        for i, b in enumerate(self.bl):
            f = b(z)
            if self.back is not None:
                # doubly residual: блок вычитает свой backcast из входа, дальше идёт остаток
                z = z - self.back[i](f)
                acc = acc + f
            else:
                z = z + f
        z = acc if self.back is not None else z
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

class Model(nn.Module):
    def __init__(self, arm, d=128, dctx=64):
        super().__init__()
        self.use_vsn = arm in ("vsn", "all")
        self.use_st = arm in ("static", "all")
        self.stat = GRN(FS, 256, dctx) if self.use_st else None
        self.vsn = VSN(CH, 16, dctx if self.use_st else None) if self.use_vsn else None
        self.trunk = Trunk(16 if self.use_vsn else CH, nbeats=(arm == "all"))
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        din = self.trunk.out + 10 + 16 + (dctx if self.use_st else 0)
        self.net = nn.Sequential(nn.Linear(din, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                 nn.Linear(d, 3))
    def forward(self, x, cal, st=None):
        c = self.stat(st) if self.use_st else None
        if self.use_vsn:
            x, _ = self.vsn(x, c)
        h = self.trunk(x)
        B = h.shape[0]
        parts = [h.unsqueeze(1).expand(B, H, -1),
                 cal.unsqueeze(0).expand(B, H, -1),
                 self.pos.unsqueeze(0).expand(B, H, -1)]
        if c is not None:
            parts.append(c.unsqueeze(1).expand(B, H, -1))
        o = self.net(torch.cat(parts, -1))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

LOSS_KIND = "gauss"

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    s = ls.exp()
    if LOSS_KIND == "laplace":
        val = torch.abs(ly - mu) / s + ls          # оптимум — условная медиана
    elif LOSS_KIND == "student":
        nu = 4.0
        val = 0.5 * (nu + 1) * torch.log1p(((ly - mu) / s) ** 2 / nu) + ls
    else:
        val = 0.5 * (((ly - mu) / s) ** 2 + 2 * ls)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * val).sum(1).mean()

NQT = 128                                   # узлов по t; при 32 ошибка 1,7e-2
_gl_x, _gl_w = np.polynomial.legendre.leggauss(NQT)
_u = 0.5*(_gl_x + 1)*(14.0 - (-32.0)) + (-32.0)
T_NODES = np.exp(_u)
T_W = _gl_w * 0.5*(14.0 - (-32.0)) * T_NODES        # вес с якобианом dt = t du

def quad_readout(hz, mu, ls, gh_nodes, gh_wts, sigma, scale,
                 tau=0.0, blen=0, v_gh=5):
    """E[log1p S] точной квадратурой. tau>0 и blen>0 включают блочную frailty."""
    dev = hz.device
    t = torch.tensor(T_NODES, dtype=torch.float64, device=dev)
    tw = torch.tensor(T_W, dtype=torch.float64, device=dev)
    zz, ww = np.polynomial.hermite_e.hermegauss(24)
    zz = torch.tensor(zz, dtype=torch.float64, device=dev)
    ww = torch.tensor(ww / ww.sum(), dtype=torch.float64, device=dev)
    hz64, mu64, sd64 = hz.double(), mu.double(), ls.double().exp()
    B, Hh = hz64.shape
    # преобразование Лапласа величины чека от frailty не зависит — считаем один раз
    m_ = mu64.unsqueeze(-1).unsqueeze(-1)
    s_ = sd64.unsqueeze(-1).unsqueeze(-1)
    yv = torch.expm1(m_ + s_ * zz.view(1, 1, -1, 1)).clamp(min=0.0)
    lt = (torch.exp(-t.view(1, 1, 1, -1) * yv) * ww.view(1, 1, -1, 1)).sum(2)   # (B,H,NT)
    use_block = tau > 0 and blen > 0 and Hh % blen == 0
    if use_block:
        vz, vw = np.polynomial.hermite_e.hermegauss(v_gh)
        vw = vw / vw.sum()
        nb = Hh // blen
    acc = torch.zeros(B, dtype=torch.float64, device=dev)
    for zn, wn in zip(gh_nodes, gh_wts):
        if not use_block:
            p = torch.sigmoid(hz64 * scale + float(zn) * sigma)
            fac = (1.0 - p).unsqueeze(-1) + p.unsqueeze(-1) * lt
            LS = torch.exp(torch.log(fac.clamp(min=1e-300)).sum(1))
        else:
            blockL = torch.zeros(B, nb, t.shape[0], dtype=torch.float64, device=dev)
            for vn, vwt in zip(vz, vw):
                p = torch.sigmoid(hz64 * scale + float(zn) * sigma + float(vn) * tau)
                fac = (1.0 - p).unsqueeze(-1) + p.unsqueeze(-1) * lt
                lf = torch.log(fac.clamp(min=1e-300)).view(B, nb, blen, -1).sum(2)
                blockL = blockL + float(vwt) * torch.exp(lf)
            LS = torch.exp(torch.log(blockL.clamp(min=1e-300)).sum(1))
        acc += float(wn) * ((1.0 - LS) * torch.exp(-t) / t * tw).sum(-1)
    return acc

def mixture_nodes(gh, sigma, w2, shift, sig2):
    """узлы и веса смеси двух логнормалей по frailty: (1-w2) при sigma и w2 при sig2 со сдвигом"""
    n1, a1 = np.polynomial.hermite_e.hermegauss(gh)
    a1 = a1 / a1.sum()
    z1 = n1 * sigma
    z2 = n1 * sig2 + shift
    return (np.concatenate([z1, z2]),
            np.concatenate([a1 * (1 - w2), a1 * w2]))

def fit_cache(arm, seed=0, TR=None, EV=None):
    """обучает модель один раз и возвращает её выходы на якоре оценки"""
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Model(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in TR}
    for _ in range(EPOCHS):
        m.train()
        for a in TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                st = torch.from_numpy(ST[a][idx]).to(DEV) if m.use_st else None
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a], st), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    m.eval()
    cal = torch.from_numpy(cal_feats(EV)).to(DEV)
    HZ = np.empty((NU, H), np.float32); MU = np.empty((NU, H), np.float32)
    LS = np.empty((NU, H), np.float32)
    with torch.no_grad():
        for i in range(0, NU, 512):
            idx = np.arange(i, min(i + 512, NU))
            st = torch.from_numpy(ST[EV][idx]).to(DEV) if m.use_st else None
            hz, mu, ls = m(torch.from_numpy(win(idx, EV)).to(DEV), cal, st)
            HZ[idx] = hz.float().cpu().numpy(); MU[idx] = mu.float().cpu().numpy()
            LS[idx] = ls.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return HZ, MU, LS

def readout(HZ, MU, LS, nodes, wts, sigma, scale, tau=0.0, blen=0, chunk=256):
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, chunk):
            j = min(i + chunk, NU)
            hz = torch.from_numpy(HZ[i:j]).to(DEV)
            mu = torch.from_numpy(MU[i:j]).to(DEV)
            ls = torch.from_numpy(LS[i:j]).to(DEV)
            out[i:j] = quad_readout(hz, mu, ls, nodes, wts, sigma, scale,
                                    tau=tau, blen=blen).float().cpu().numpy()
    return out

y = target(SEL_EV); ly = np.log1p(y); MU_L = ly.mean()
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU_L, 0, 13)))

K = int(os.environ.get("K", 5))
ITERS = int(os.environ.get("ITERS", 25))
ANCH = [138, 168, 198, 228, 258, 288, 318]     # по 30 дней каждый: дни 139..348
CHU = 20000     # размер куска для EM; CH в заголовке — это ЧИСЛО КАНАЛОВ (11)

print(f"обучаю дневную модель один раз ({EPOCHS} эпох) ...", flush=True)
t0 = time.time()
torch.manual_seed(0); rng = np.random.default_rng(0)
m = Model("all").to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
steps = EPOCHS * len(SEL_TR) * (NU // BS)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in SEL_TR}
for _ in range(EPOCHS):
    m.train()
    for a in SEL_TR:
        perm = rng.permutation(NU)
        for i0 in range(0, NU, BS):
            idx = np.sort(perm[i0:i0 + BS])
            if len(idx) < 64:
                continue
            x = torch.from_numpy(win(idx, a)).to(DEV)
            st = torch.from_numpy(ST[a][idx]).to(DEV) if m.use_st else None
            yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss_fn(*m(x, CAL[a], st), yd).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            if sch.last_epoch < steps - 1: sch.step()
m.eval()
print(f"  обучено за {time.time()-t0:.0f}s; снимаю поденные выходы на {len(ANCH)+1} якорях ...",
      flush=True)

def snap(a):
    cal = torch.from_numpy(cal_feats(a)).to(DEV)
    HZ = np.empty((NU, H), np.float32); MU = np.empty((NU, H), np.float32)
    LS = np.empty((NU, H), np.float32)
    with torch.no_grad():
        for i0 in range(0, NU, 512):
            idx = np.arange(i0, min(i0 + 512, NU))
            st = torch.from_numpy(ST[a][idx]).to(DEV) if m.use_st else None
            hz, mu_, ls_ = m(torch.from_numpy(win(idx, a)).to(DEV), cal, st)
            HZ[idx] = hz.float().cpu().numpy(); MU[idx] = mu_.float().cpu().numpy()
            LS[idx] = ls_.float().cpu().numpy()
    return HZ, MU, LS

parts = [snap(a) for a in ANCH]
HZh = np.concatenate([p[0] for p in parts], 1)
MUh = np.concatenate([p[1] for p in parts], 1)
LSh = np.concatenate([p[2] for p in parts], 1)
HZe, MUe, LSe = snap(SEL_EV)
del parts
d0 = ANCH[0] + 1
Zh = (G[:, d0:d0 + HZh.shape[1]] > 0)
LGh = np.where(Zh, np.log1p(G[:, d0:d0 + HZh.shape[1]]), 0.0).astype(np.float32)
print(f"  история {Zh.shape}, доля активных {Zh.mean():.4f}", flush=True)
del m; torch.cuda.empty_cache()

a_s = torch.linspace(-0.8, 0.8, K, device=DEV).double()
m_s = torch.zeros(K, device=DEV).double()
v_s = torch.ones(K, device=DEV).double()
A = torch.eye(K, device=DEV).double() * 6.0 + 1.0
A = A / A.sum(1, keepdim=True)
p0 = torch.full((K,), 1.0 / K, device=DEV).double()

def emis(hz, mu_, sd_, zc, lgc):
    lo = hz.unsqueeze(-1) + a_s.view(1, 1, -1)
    lp1 = -torch.nn.functional.softplus(-lo)
    lp0 = -torch.nn.functional.softplus(lo)
    sdv = sd_.unsqueeze(-1) * v_s.view(1, 1, -1)
    x = (lgc.unsqueeze(-1) - mu_.unsqueeze(-1) - m_s.view(1, 1, -1)) / sdv
    lpa = -0.5 * x * x - torch.log(sdv) - 0.9189385332
    return torch.where(zc.unsqueeze(-1), lp1 + lpa, lp0), lo

def fwdbwd(le):
    B, Tt, _ = le.shape
    lA = torch.log(A).unsqueeze(0)
    al = torch.empty(B, Tt, K, dtype=torch.float64, device=DEV)
    al[:, 0] = torch.log(p0).unsqueeze(0) + le[:, 0]
    for t_ in range(1, Tt):
        al[:, t_] = torch.logsumexp(al[:, t_ - 1].unsqueeze(-1) + lA, 1) + le[:, t_]
    be = torch.zeros(B, Tt, K, dtype=torch.float64, device=DEV)
    for t_ in range(Tt - 2, -1, -1):
        be[:, t_] = torch.logsumexp(lA + (le[:, t_ + 1] + be[:, t_ + 1]).unsqueeze(1), 2)
    ll = torch.logsumexp(al[:, -1], 1)
    gam = torch.softmax(al + be, dim=-1)
    xi = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    for t_ in range(Tt - 1):
        mm = al[:, t_].unsqueeze(-1) + lA + (le[:, t_ + 1] + be[:, t_ + 1]).unsqueeze(1)
        xi += torch.exp(mm - ll.view(-1, 1, 1)).sum(0)
    return gam, xi, ll.sum()

print(f"\n{'итерация':>9} {'логправдоподобие':>18} {'сек':>7}", flush=True)
for it in range(ITERS):
    t0 = time.time()
    nx = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    g1 = torch.zeros(K, dtype=torch.float64, device=DEV)
    g2 = torch.zeros(K, dtype=torch.float64, device=DEV)
    sw = torch.zeros(K, dtype=torch.float64, device=DEV)
    sx = torch.zeros(K, dtype=torch.float64, device=DEV)
    sx2 = torch.zeros(K, dtype=torch.float64, device=DEV)
    sp0 = torch.zeros(K, dtype=torch.float64, device=DEV)
    tot = 0.0
    for i0 in range(0, NU, CHU):
        j0 = min(i0 + CHU, NU)
        hz = torch.from_numpy(HZh[i0:j0]).to(DEV).double()
        mu_ = torch.from_numpy(MUh[i0:j0]).to(DEV).double()
        sd_ = torch.from_numpy(LSh[i0:j0]).to(DEV).double().exp()
        zc = torch.from_numpy(Zh[i0:j0]).to(DEV)
        lgc = torch.from_numpy(LGh[i0:j0]).to(DEV).double()
        le, lo = emis(hz, mu_, sd_, zc, lgc)
        gam, xi, ll = fwdbwd(le)
        nx += xi; tot += float(ll); sp0 += gam[:, 0].sum(0)
        sig = torch.sigmoid(lo)
        g1 += (gam * (zc.unsqueeze(-1).double() - sig)).sum((0, 1))
        g2 += (gam * sig * (1.0 - sig)).sum((0, 1))
        gz = gam * zc.unsqueeze(-1)
        d = ((lgc - mu_) / sd_).unsqueeze(-1)
        sw += gz.sum((0, 1)); sx += (gz * d).sum((0, 1)); sx2 += (gz * d * d).sum((0, 1))
    A = (nx / nx.sum(1, keepdim=True)).clamp(1e-6, 1.0); A = A / A.sum(1, keepdim=True)
    p0 = (sp0 / sp0.sum()).clamp(1e-6, 1.0)
    a_s = a_s + g1 / g2.clamp(min=1e-9)
    mstd = sx / sw.clamp(min=1e-9)
    m_s = mstd * 0.0 + mstd            # сдвиг в единицах sigma модели
    v_s = ((sx2 / sw.clamp(min=1e-9)) - mstd * mstd).clamp(min=0.04).sqrt()
    print(f"{it:>9} {tot/NU:18.5f} {time.time()-t0:7.0f}", flush=True)

print(f"\nсдвиг логита a={np.round(a_s.cpu().numpy(),3)}")
print(f"сдвиг чека m={np.round(m_s.cpu().numpy(),3)}  масштаб v={np.round(v_s.cpu().numpy(),3)}")
dg = torch.diag(A).cpu().numpy()
print(f"диагональ A: {np.round(dg,4)}  время автокорреляции {1.0/(1.0-float(dg.mean())):.2f} дня")

NQT = 128
_gx, _gw = np.polynomial.legendre.leggauss(NQT)
_u = 0.5 * (_gx + 1) * (14.0 - (-32.0)) + (-32.0)
TN = torch.tensor(np.exp(_u), dtype=torch.float64, device=DEV)
TW = torch.tensor(_gw * 0.5 * (14.0 - (-32.0)) * np.exp(_u), dtype=torch.float64, device=DEV)
zz, ww = np.polynomial.hermite_e.hermegauss(24)
zz = torch.tensor(zz, dtype=torch.float64, device=DEV)
ww = torch.tensor(ww / ww.sum(), dtype=torch.float64, device=DEV)

print("\nсчитываю горизонт прямым ходом ...", flush=True)
out = np.empty(NU, np.float32); t0 = time.time()
with torch.no_grad():
    for i0 in range(0, NU, 2000):
        j0 = min(i0 + 2000, NU)
        hz = torch.from_numpy(HZh[i0:j0]).to(DEV).double()
        mu_ = torch.from_numpy(MUh[i0:j0]).to(DEV).double()
        sd_ = torch.from_numpy(LSh[i0:j0]).to(DEV).double().exp()
        zc = torch.from_numpy(Zh[i0:j0]).to(DEV)
        lgc = torch.from_numpy(LGh[i0:j0]).to(DEV).double()
        le, _ = emis(hz, mu_, sd_, zc, lgc)
        lA = torch.log(A).unsqueeze(0)
        al = torch.log(p0).unsqueeze(0) + le[:, 0]
        for t_ in range(1, le.shape[1]):
            al = torch.logsumexp(al.unsqueeze(-1) + lA, 1) + le[:, t_]
        post = torch.softmax(al, dim=-1)
        hze = torch.from_numpy(HZe[i0:j0]).to(DEV).double()
        mue = torch.from_numpy(MUe[i0:j0]).to(DEV).double()
        sde = torch.from_numpy(LSe[i0:j0]).to(DEV).double().exp()
        al2 = post.unsqueeze(-1).expand(-1, -1, NQT).clone()
        for d_ in range(H):
            pe = torch.sigmoid(hze[:, d_].unsqueeze(-1) + a_s.view(1, -1))         # (B,K)
            sdv = sde[:, d_].unsqueeze(-1) * v_s.view(1, -1)
            yv = torch.expm1(mue[:, d_].unsqueeze(-1).unsqueeze(-1)
                             + m_s.view(1, -1, 1) * sde[:, d_].view(-1, 1, 1)
                             + sdv.unsqueeze(-1) * zz.view(1, 1, -1)).clamp(min=0.0)
            lam = (torch.exp(-TN.view(1, 1, 1, -1) * yv.unsqueeze(-1))
                   * ww.view(1, 1, -1, 1)).sum(2)
            Lt = (1.0 - pe).unsqueeze(-1) + pe.unsqueeze(-1) * lam
            al2 = torch.einsum("bkt,ks->bst", al2, A) * Lt
        LS = al2.sum(1)
        out[i0:j0] = (((1.0 - LS) * torch.exp(-TN) / TN * TW).sum(-1)).float().cpu().numpy()
print(f"  готово за {time.time()-t0:.0f}s", flush=True)
np.save(OUT / f"x8_hmmcov_K{K}.npy", out)
print(f"\nHMM поверх дневной модели, K={K}: RMSLE {sc(out):.5f}  -> x8_hmmcov_K{K}.npy")
for nm in ("h8_gauss", "q2c_daily_sel"):
    f = OUT / f"{nm}.npy"
    if f.exists():
        r = np.load(f)
        if r.shape == (NU,):
            print(f"  corr с {nm}: {np.corrcoef(out, r)[0,1]:.5f}")
print("\nвклад считается отдельно: w11 с WHO=x8_hmmcov_K5")
print("\ndone")
