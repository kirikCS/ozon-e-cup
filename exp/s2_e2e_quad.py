"""Обучение головы ПРЯМО ПОД МЕТРИКУ через квадратуру.

Сейчас голова учится на подённом правдоподобии: BCE на факт покупки плюс NLL логнормали на
величину. Это НЕ метрика. Зазор между «хорошо описывает дни» и «хорошо предсказывает
E[log1p(сумма за 30 дней)]» систематический: ошибки в sigma и в дисперсии frailty сдвигают
месячный логарифм смещённо, а подённое правдоподобие их в нужную сторону не штрафует.

p9 показал, что квадратура Фруллани точно воспроизводит предел Монте-Карло (sd разности падает
как 1/sqrt(n) с отношениями 2,005 / 2,001 / 1,992). Но главное её свойство другое:
она ДИФФЕРЕНЦИРУЕМА, а симуляция путей — нет. Значит цепь замыкается:

    признаки -> (pi_d, m_d, sigma_d) -> квадратура -> MSE против log1p(y) -> назад

Рецепт классический и здесь применим целиком: сначала предобучение на правдоподобии (оно даёт
осмысленную инициализацию всех трёх голов), затем дообучение под метрику. Обучать сразу под
метрику нельзя — у хазарда и логнормали нет градиента, пока предсказание не в правильной области.

Стоимость замерена: тензор severity-LT на батч 8,8М элементов (35 МБ fp32), цикл по узлам
frailty последовательный, 14 мс на шаг, плюс пять минут к фазе. При обучении сетка грубее
(48 узлов по t, 12 по severity, 8 по frailty), на выводе — полная (128/24/16).

Контроль — та же модель после фазы 1 со считыванием квадратурой, то есть ровно p9/r6.
"""
import sys
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

def quad_readout(hz, mu, ls, gh_nodes, gh_wts, sigma, scale):
    """E[log1p S] точной квадратурой. hz, mu, ls: (B, H). Возвращает (B,)"""
    dev = hz.device
    t = torch.tensor(T_NODES, dtype=torch.float64, device=dev)          # (NT,)
    tw = torch.tensor(T_W, dtype=torch.float64, device=dev)
    zz, ww = np.polynomial.hermite_e.hermegauss(24)
    zz = torch.tensor(zz, dtype=torch.float64, device=dev)
    ww = torch.tensor(ww / ww.sum(), dtype=torch.float64, device=dev)
    hz64, mu64, sd64 = hz.double(), mu.double(), ls.double().exp()
    acc = torch.zeros(hz.shape[0], dtype=torch.float64, device=dev)
    for zn, wn in zip(gh_nodes, gh_wts):
        p = torch.sigmoid(hz64*scale + float(zn)*sigma)                 # (B,H)
        # E_LN[exp(-t*y)], y = expm1(m + sd*Z) >= 0
        m = mu64.unsqueeze(-1).unsqueeze(-1)                            # (B,H,1,1)
        s = sd64.unsqueeze(-1).unsqueeze(-1)
        yv = torch.expm1(m + s*zz.view(1,1,-1,1)).clamp(min=0.0)        # (B,H,GH,1)
        lt = (torch.exp(-t.view(1,1,1,-1)*yv) * ww.view(1,1,-1,1)).sum(2)   # (B,H,NT)
        fac = (1.0 - p).unsqueeze(-1) + p.unsqueeze(-1)*lt              # (B,H,NT)
        LS = torch.exp(torch.log(fac.clamp(min=1e-300)).sum(1))         # (B,NT)
        acc += float(wn) * ((1.0 - LS) * torch.exp(-t) / t * tw).sum(-1)
    return acc

NQT_TR, GHS_TR, GHF_TR = 48, 12, 8
_gx, _gw = np.polynomial.legendre.leggauss(NQT_TR)
_ut = 0.5*(_gx + 1)*(14.0 - (-32.0)) + (-32.0)
T_TR = np.exp(_ut); TW_TR = _gw * 0.5*(14.0 - (-32.0)) * T_TR

def quad_train(hz, mu, ls, sigma, scale, dev):
    """дифференцируемая квадратура пониженного разрешения — для обучающего шага"""
    t = torch.tensor(T_TR, dtype=torch.float32, device=dev)
    tw = torch.tensor(TW_TR, dtype=torch.float32, device=dev)
    zs, ws = np.polynomial.hermite_e.hermegauss(GHS_TR)
    zs = torch.tensor(zs, dtype=torch.float32, device=dev)
    ws = torch.tensor(ws/ws.sum(), dtype=torch.float32, device=dev)
    zf, wf = np.polynomial.hermite_e.hermegauss(GHF_TR)
    wf = wf/wf.sum()
    sd = ls.exp()
    acc = torch.zeros(hz.shape[0], device=dev)
    for zn, wn in zip(zf, wf):
        p = torch.sigmoid(hz*scale + float(zn)*sigma)                    # (B,H)
        yv = torch.expm1(mu.unsqueeze(-1).unsqueeze(-1)
                         + sd.unsqueeze(-1).unsqueeze(-1)*zs.view(1,1,-1,1)).clamp(min=0.0)
        lt = (torch.exp(-t.view(1,1,1,-1)*yv) * ws.view(1,1,-1,1)).sum(2)  # (B,H,NT)
        fac = (1.0 - p).unsqueeze(-1) + p.unsqueeze(-1)*lt
        LS = torch.exp(torch.log(fac.clamp(min=1e-30)).sum(1))             # (B,NT)
        acc = acc + float(wn) * ((1.0 - LS) * torch.exp(-t) / t * tw).sum(-1)
    return acc

def run(arm, seed=0, e2e=False, TR=None, EV=None):
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
                if len(idx) < 64: continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                st = torch.from_numpy(ST[a][idx]).to(DEV) if m.use_st else None
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a], st), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    if e2e:
        # ФАЗА 2: дообучение под метрику через квадратуру
        scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
        opt2 = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-5)
        st2 = E2EPOCHS * len(TR) * (NU // BS)
        sc2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(st2, 10))
        for ep in range(E2EPOCHS):
            m.train(); tot = 0.0; nb = 0
            for a in TR:
                perm = rng.permutation(NU)
                for i in range(0, NU, BS):
                    idx = np.sort(perm[i:i + BS])
                    if len(idx) < 64: continue
                    x = torch.from_numpy(win(idx, a)).to(DEV)
                    stt = torch.from_numpy(ST[a][idx]).to(DEV) if m.use_st else None
                    yt = torch.from_numpy(
                        np.log1p(G[idx, a + 1:a + 1 + H].astype(np.float64).sum(1))
                    ).float().to(DEV)
                    opt2.zero_grad(set_to_none=True)
                    hz, mu, ls = m(x, CAL[a], stt)
                    pred = quad_train(hz, mu, ls, SIGMA, scale, DEV)
                    loss = ((pred - yt) ** 2).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt2.step()
                    if sc2.last_epoch < st2 - 1: sc2.step()
                    tot += float(loss); nb += 1
            print(f"    фаза 2, эпоха {ep}: MSE {tot/max(nb,1):.5f} "
                  f"(RMSE {np.sqrt(tot/max(nb,1)):.5f})", flush=True)
    m.eval()
    cal = torch.from_numpy(cal_feats(EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    PATHS = [96, 384, 1536, 6144]
    out = {n: np.empty(NU, np.float32) for n in PATHS}
    out["quad"] = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            st = torch.from_numpy(ST[EV][idx]).to(DEV) if m.use_st else None
            hz, mu, ls = m(torch.from_numpy(win(idx, EV)).to(DEV), cal, st)
            out["quad"][idx] = quad_readout(hz, mu, ls, nodes, wts, SIGMA, scale
                                            ).float().cpu().numpy()
            hzs = hz * scale; sd = ls.exp()
            for npath in PATHS:
                acc = torch.zeros(len(idx), device=DEV)
                for zn, wn in zip(nodes, wts):
                    p = torch.sigmoid(hzs + float(zn) * SIGMA)
                    u = torch.rand(npath, len(idx), H, device=DEV) < p.unsqueeze(0)
                    v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                    * torch.randn(npath, len(idx), H, device=DEV))
                    acc += float(wn) * torch.log1p((u.float()*v.clamp(min=0)).sum(-1)).mean(0)
                out[npath][idx] = acc.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out

E2EPOCHS = int(__import__("os").environ.get("E2EPOCHS", 2))
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))
print(f"\n=== обучение под метрику через квадратуру (tft_all, сид 0) ===", flush=True)
print(f"фаза 1: подённое правдоподобие, {EPOCHS} эпох; "
      f"фаза 2: MSE через квадратуру, {E2EPOCHS} эпох", flush=True)
print(f"\n{'рука':>6} {'RMSLE':>10} {'сек':>7}", flush=True)
RES = {}
for nm, flag in (("nll", False), ("e2e", True)):
    t0 = time.time()
    R = run("all", seed=0, e2e=flag)
    v = R["quad"]
    RES[nm] = v
    np.save(OUT / f"s2_{nm}.npy", v)
    print(f"{nm:>6} {sc(v):10.5f} {time.time()-t0:7.0f}", flush=True)
if len(RES) == 2:
    print(f"\ncorr(nll, e2e) = {np.corrcoef(RES['nll'], RES['e2e'])[0,1]:.5f}")
    print(f"дельта: {sc(RES['e2e']) - sc(RES['nll']):+.2e}")
print("\ndone")
