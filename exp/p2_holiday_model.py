"""Явные праздничные признаки в горизонте: отклик переносится между праздниками?

Форма вокруг прошлогодних праздников измерена в p1 и отчётлива: подъём 5-20% за неделю до,
провал в сам день (23.02 - GMV 0,953 при активности 0,936; 08.03 - 0,942 при 0,854), возврат
за 2-3 дня. Чистый прирост за 30 дней +5,6% против 0,984-0,991 у нейтральных окон.

Ключевое наблюдение: новогоднее окно даёт 1,0500, почти тот же прирост, что праздничное 1,0556.
Значит отклик переносим между праздниками. А в cal_feats флага праздника НЕТ вовсе - семь
одномерных дня недели, день/31 и синус-косинус дня года. Гладкая годовая сезонность спайк на
23 февраля не представляет физически.

Протокол устроен так, что вопрос ставится честно и даже строже реального. Оценочный якорь 348
имеет горизонт 349..378 - это новогоднее окно. Горизонты обучающих якорей SEL_TR (207..236,
235..264, 263..292, 291..320, 319..348) крупных праздников не содержат. То есть модель обязана
ЭКСТРАПОЛИРОВАТЬ отклик на праздник, которого в обучении не видела. В реальном финале легче:
FIN_TR содержит анкер 350, чей горизонт покрывает Новый год.

Добавляется 6 колонок к календарю горизонта: флаг праздника, канун (1..7 дней до, линейно),
хвост (1..3 дня после), флаг новогодних каникул, и отдельно расстояние до ближайшего праздника
вперёд и назад. Две руки, один сид, всё остальное неизменно.
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

from datetime import date as _date

HOLIDAYS = [
    _date(2025,1,1), _date(2025,1,7), _date(2025,2,23), _date(2025,3,8),
    _date(2025,5,1), _date(2025,5,9), _date(2025,6,12), _date(2025,11,4),
    _date(2025,12,31),
    _date(2026,1,1), _date(2026,1,7), _date(2026,2,23), _date(2026,3,8),
]
NY = [(_date(2024,12,29), _date(2025,1,8)), (_date(2025,12,29), _date(2026,1,8))]
USE_HOL = False
NHOL = 6

def hol_feats(a):
    out = np.zeros((H, NHOL), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=a + 1 + i)
        dd = [(h - d).days for h in HOLIDAYS]
        fwd = [x for x in dd if x >= 0]
        bwd = [-x for x in dd if x <= 0]
        nf = min(fwd) if fwd else 99
        nb = min(bwd) if bwd else 99
        out[i, 0] = 1.0 if nf == 0 else 0.0                 # сам праздник
        out[i, 1] = max(0.0, (8 - nf) / 7.0) if nf > 0 else 0.0   # канун, линейный подъём
        out[i, 2] = max(0.0, (4 - nb) / 3.0) if nb > 0 else 0.0   # хвост после
        out[i, 3] = 1.0 if any(lo <= d <= hi for lo, hi in NY) else 0.0
        out[i, 4] = min(nf, 30) / 30.0
        out[i, 5] = min(nb, 30) / 30.0
    return out

_cal_plain = cal_feats
def cal_feats(a):
    c = _cal_plain(a)
    return np.concatenate([c, hol_feats(a)], 1) if USE_HOL else c

NCAL = 10 + (NHOL if USE_HOL else 0)

class Model(nn.Module):
    def __init__(self, arm, d=128, dctx=64):
        super().__init__()
        self.use_vsn = arm in ("vsn", "all")
        self.use_st = arm in ("static", "all")
        self.stat = GRN(FS, 256, dctx) if self.use_st else None
        self.vsn = VSN(CH, 16, dctx if self.use_st else None) if self.use_vsn else None
        self.trunk = Trunk(16 if self.use_vsn else CH, nbeats=(arm == "all"))
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        din = self.trunk.out + NCAL + 16 + (dctx if self.use_st else 0)
        self.net = nn.Sequential(nn.Linear(din, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                 nn.Linear(d, 3))
    def forward(self, x, cal, st=None):
        c = self.stat(st) if self.use_st else None
        if self.use_vsn:
            x, _ = self.vsn(x, c)
        h = self.trunk(x)
        B = h.shape[0]
        parts = [h.unsqueeze(1).expand(B, H, -1), cal.unsqueeze(0).expand(B, H, -1),
                 self.pos.unsqueeze(0).expand(B, H, -1)]
        if c is not None:
            parts.append(c.unsqueeze(1).expand(B, H, -1))
        o = self.net(torch.cat(parts, -1))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(arm, seed=0, TR=None, EV=None):
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
    m.eval()
    cal = torch.from_numpy(cal_feats(EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32); daily = np.zeros(H, np.float64)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            st = torch.from_numpy(ST[EV][idx]).to(DEV) if m.use_st else None
            hz, mu, ls = m(torch.from_numpy(win(idx, EV)).to(DEV), cal, st)
            daily += torch.sigmoid(hz).sum(0).double().cpu().numpy()
            hz = hz * scale; sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NPATH, len(idx), H, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out, daily / NU

y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
fact = (G[:, SEL_EV + 1:SEL_EV + 1 + H] > 0).mean(0)
print(f"\n=== явные праздничные признаки, горизонт якоря {SEL_EV} = "
      f"{D0+timedelta(days=SEL_EV+1)} .. {D0+timedelta(days=SEL_EV+H)} ===", flush=True)
print(f"{'рука':>10} {'RMSLE':>10} {'сек':>7} {'corr дневной формы с фактом':>30}", flush=True)
RES = {}
for tag, flag in (("без праздников", False), ("с праздниками", True)):
    globals()["USE_HOL"] = flag
    globals()["NCAL"] = 10 + (NHOL if flag else 0)
    t0 = time.time()
    v, dly = run("all", seed=0)
    RES[tag] = v
    sh = v - v.mean() + MU
    print(f"{tag:>10} {rmsle(y, np.expm1(np.clip(sh,0,13))):10.5f} {time.time()-t0:7.0f} "
          f"{np.corrcoef(dly, fact)[0,1]:30.4f}", flush=True)
    np.save(OUT / f"p2_{'hol' if flag else 'plain'}.npy", v)
print(f"\ncorr(без, с) = {np.corrcoef(RES['без праздников'], RES['с праздниками'])[0,1]:.5f}")
print("\ndone")
