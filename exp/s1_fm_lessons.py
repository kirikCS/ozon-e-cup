"""Что берём из архитектур foundation-моделей — две правки, найденные по их исходникам.

Обе — не украшения, а закрытие дыр, которые в наших моделях есть, а у них закрыты явно.

ПРАВКА 1: МАСКА ПАДДИНГА (из PatchTST-FM).
Наш win() дополняет окно слева нулями, когда оно уходит за начало истории, и модель видит их
как «клиент не покупал». Масштаб: при L=364 на якоре 206 это 157 дней = 43,1% окна, на 234 —
35,4%, на 318 — 12,4%, на 348 — 4,1%, на 408 — ноль. То есть доля фальшивых нулей
СИСТЕМАТИЧЕСКИ меняется между обучением и выводом, и распознать это модель не может.

PatchTST-FM несёт три отдельные маски (pred, pad, miss) и КОНКАТЕНИРУЕТ ~mask ко входу
(`in_layer(cat([x_patch, ~mask]))`), а не просто зануляет. Здесь добавляется один канал:
1 где данные настоящие, 0 где паддинг.

ПРАВКА 2: VSN БЕЗ SOFTMAX (из HSTU).
Наш VSN нормирует веса каналов softmax'ом, то есть в сумму 1. Формулировка авторов HSTU:
«the number of prior data points related to target serves as a strong feature indicating the
intensity of user preferences, which is hard to capture after the softmax normalization».
При 94% нулевых дней абсолютная интенсивность — вероятно, главный признак, а softmax стирает
её на каждом слое. Замена на сигмоиду делает каналы независимыми гейтами.

Четыре руки, один сид, всё остальное неизменно. Контроль — базовая рука тем же кодом.
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


USE_MASK = False
NOSOFTMAX = False

def win(users, end):
    s = end - L + 1
    if s >= 0:
        x = SEQ[users, :, s:end + 1].astype(np.float32)
        m = np.ones((len(users), 1, L), np.float32)
    else:
        got = SEQ[users, :, 0:end + 1].astype(np.float32)
        pad = np.zeros((len(users), CH, -s), np.float32)
        x = np.concatenate([pad, got], 2)
        m = np.concatenate([np.zeros((len(users), 1, -s), np.float32),
                            np.ones((len(users), 1, end + 1), np.float32)], 2)
    return np.concatenate([x, m], 1) if USE_MASK else x

class VSN(nn.Module):
    """по каждому шагу времени: softmax по каналам (веса в сумму 1) либо независимые сигмоидные
    гейты (абсолютная интенсивность сохраняется) — переключается NOSOFTMAX"""
    def __init__(self, nvar, d, dctx=None):
        super().__init__()
        self.per = nn.ModuleList([GRN(1, d, d) for _ in range(nvar)])
        self.sel = GRN(nvar, d, nvar, dctx=dctx)
        self.nvar = nvar
    def forward(self, x, c=None):
        xt = x.transpose(1, 2)
        raw = self.sel(xt, c.unsqueeze(1).expand(-1, xt.shape[1], -1) if c is not None else None)
        w = torch.sigmoid(raw) if NOSOFTMAX else torch.softmax(raw, -1)
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
                z = z - self.back[i](f); acc = acc + f
            else:
                z = z + f
        z = acc if self.back is not None else z
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

class Model(nn.Module):
    def __init__(self, arm, d=128, dctx=64):
        super().__init__()
        nch = CH + (1 if USE_MASK else 0)
        self.use_vsn = arm in ("vsn", "all")
        self.use_st = arm in ("static", "all")
        self.stat = GRN(FS, 256, dctx) if self.use_st else None
        self.vsn = VSN(nch, 16, dctx if self.use_st else None) if self.use_vsn else None
        self.trunk = Trunk(16 if self.use_vsn else nch, nbeats=(arm == "all"))
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        din = self.trunk.out + 10 + 16 + (dctx if self.use_st else 0)
        self.net = nn.Sequential(nn.Linear(din, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                 nn.Linear(d, 3))
    def forward(self, x, cal, st=None):
        c = self.stat(st) if self.use_st else None
        if self.use_vsn:
            x, _ = self.vsn(x, c)
        h = self.trunk(x); B = h.shape[0]
        parts = [h.unsqueeze(1).expand(B, H, -1), cal.unsqueeze(0).expand(B, H, -1),
                 self.pos.unsqueeze(0).expand(B, H, -1)]
        if c is not None:
            parts.append(c.unsqueeze(1).expand(B, H, -1))
        o = self.net(torch.cat(parts, -1))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); lyv = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyv - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

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
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            st = torch.from_numpy(ST[EV][idx]).to(DEV) if m.use_st else None
            hz, mu, ls = m(torch.from_numpy(win(idx, EV)).to(DEV), cal, st)
            hz = hz * scale; sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NPATH, len(idx), H, device=DEV))
                acc += float(wn) * torch.log1p((u.float()*v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out

import os
ONLY = os.environ.get("ARMS", "base,mask,nosm,both").split(",")
y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))
print(f"\n=== правки из foundation-моделей (tft_all, сид 0) ===", flush=True)
print(f"паддинг в окне L={L}: якорь 206 — {max(0,L-207)/L:.1%}, 318 — {max(0,L-319)/L:.1%}, "
      f"348 — {max(0,L-349)/L:.1%}, 408 — 0%", flush=True)
print(f"\n{'рука':>8} {'маска':>7} {'без softmax':>12} {'RMSLE':>10} {'сек':>7}", flush=True)
CFG = {"base": (False, False), "mask": (True, False),
       "nosm": (False, True),  "both": (True, True)}
for nm in ONLY:
    if nm not in CFG: continue
    globals()["USE_MASK"], globals()["NOSOFTMAX"] = CFG[nm]
    t0 = time.time()
    v = run("all", seed=0)
    np.save(OUT / f"s1_{nm}.npy", v)
    print(f"{nm:>8} {str(CFG[nm][0]):>7} {str(CFG[nm][1]):>12} {sc(v):10.5f} "
          f"{time.time()-t0:7.0f}", flush=True)
print("\ndone")
