"""99-квантильная голова на НАШЕМ стволе: последний открытый вопрос после n3.

В n3 квантильная голова проиграла генеративной на обоих стволах — 1,78483 против 1,78134 на
PatchTST-FM и 1,78712 против 1,78234 на случайной проекции. Но обе точки сняты на слабых стволах,
где потолок 1,78, а у нас есть свой на 1,738. И разрыв между головами СУЖАЛСЯ с ростом качества
ствола: 0,0048 на случайной проекции, 0,0035 на PatchTST-FM. Две точки — слабое основание для
экстраполяции, но проверка дешёвая, а вопрос закрывается насовсем.

Содержательно вопрос такой. Генеративная голова раскладывает месяц на 30 дней и получает 30
сигналов супервизии на клиента; квантильная видит один скаляр — месячную сумму. Зато квантильная
не имеет джессеновского разрыва вообще (он измерен, 1,0906) и представляет атом непараметрически:
в n3 в ноль схлопнулось 0,329 квантилей при истинных 0,4369 — первая квантильная модель проекта,
которая атом вообще увидела (у Toto 0,003, у Chronos 0,056, у диффузии ровно 0).

Всё остальное держится неизменным: та же архитектура tft_all, те же якоря, тот же сид, та же
нормировка статики ТОЛЬКО по обучающим якорям. Меняется одна голова. Генеративная рука здесь же
как контроль — не число из прошлого запуска, а тот же код и тот же сид.
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

NQ = 99
TAU = torch.arange(1, NQ + 1, dtype=torch.float32, device=DEV) / (NQ + 1)

class Model(nn.Module):
    def __init__(self, arm, head="gen", d=128, dctx=64):
        super().__init__()
        self.head = head
        self.use_vsn = arm in ("vsn", "all")
        self.use_st = arm in ("static", "all")
        self.stat = GRN(FS, 256, dctx) if self.use_st else None
        self.vsn = VSN(CH, 16, dctx if self.use_st else None) if self.use_vsn else None
        self.trunk = Trunk(16 if self.use_vsn else CH, nbeats=(arm == "all"))
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        din = self.trunk.out + 10 + 16 + (dctx if self.use_st else 0)
        if head == "gen":
            self.net = nn.Sequential(nn.Linear(din, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                     nn.Linear(d, 3))
        else:
            # квантильная голова смотрит на клиента целиком: календарь горизонта усредняется,
            # позиционное вложение не нужно — предсказывается один скаляр, а не 30
            self.net = nn.Sequential(nn.Linear(din - 16, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(),
                                     nn.Linear(d, NQ + 1))
    def forward(self, x, cal, st=None):
        c = self.stat(st) if self.use_st else None
        if self.use_vsn:
            x, _ = self.vsn(x, c)
        h = self.trunk(x)
        B = h.shape[0]
        if self.head == "gen":
            parts = [h.unsqueeze(1).expand(B, H, -1),
                     cal.unsqueeze(0).expand(B, H, -1),
                     self.pos.unsqueeze(0).expand(B, H, -1)]
            if c is not None:
                parts.append(c.unsqueeze(1).expand(B, H, -1))
            o = self.net(torch.cat(parts, -1))
            return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)
        parts = [h, cal.mean(0).unsqueeze(0).expand(B, -1)]
        if c is not None:
            parts.append(c)
        o = self.net(torch.cat(parts, -1))
        # монотонность тем же приёмом, что у самой PatchTST-FM: база плюс кумулятивные softplus
        return o[:, :1] + torch.cumsum(Fn.softplus(o[:, 1:]) / NQ, 1)

def gen_loss(out, yd):
    hz, mu, ls = out
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def q_loss(q, yd):
    """пинбол по 99 квантилям L = log1p(месячная сумма); считаем на несжатых значениях,
    чтобы градиент шёл и ниже нуля — обрезка только на выводе"""
    d = torch.log1p(yd.sum(1)).unsqueeze(1) - q
    return torch.maximum(TAU * d, (TAU - 1) * d).mean(1).mean()

def run(arm, head="gen", seed=0, TR=None, EV=None):
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Model(arm, head).to(DEV)
    lossf = gen_loss if head == "gen" else q_loss
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
                lossf(m(x, CAL[a], st), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    cal = torch.from_numpy(cal_feats(EV)).to(DEV)
    out = np.empty(NU, np.float32)
    zf = np.empty(NU, np.float32)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            st = torch.from_numpy(ST[EV][idx]).to(DEV) if m.use_st else None
            o = m(torch.from_numpy(win(idx, EV)).to(DEV), cal, st)
            if head == "gen":
                hz, mu, ls = o
                hz = hz * scale
                sd = ls.exp()
                acc = torch.zeros(len(idx), device=DEV)
                for zn, wn in zip(nodes, wts):
                    p = torch.sigmoid(hz + float(zn) * SIGMA)
                    u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                    v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                    * torch.randn(NPATH, len(idx), H, device=DEV))
                    acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
                out[idx] = acc.float().cpu().numpy()
                zf[idx] = np.nan
            else:
                # E[log1p Y] = интеграл log1p по подогнанной квантильной функции
                q = o.clamp(min=0)
                out[idx] = q.mean(1).float().cpu().numpy()
                zf[idx] = (q < 1e-3).float().mean(1).cpu().numpy()
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return out, npar, float(np.nanmean(zf)) if head != "gen" else None

y = target(SEL_EV)
ly = np.log1p(y)
print(f"\nдоля нулей в цели: {(y == 0).mean():.4f}", flush=True)
print(f"\n=== tft_all: генеративная голова против 99-квантильной ===", flush=True)
print(f"{'голова':>9} {'параметров':>11} {'RMSLE':>9} {'нулей':>7} {'сек':>7}", flush=True)
RES = {}
for head in ("gen", "q99"):
    t0 = time.time()
    v, npar, zf = run("all", head=head, seed=0)
    RES[head] = v
    sh = v - v.mean() + ly.mean()
    sc = rmsle(y, np.expm1(np.clip(sh, 0, 13)))
    print(f"{head:>9} {npar:11d} {sc:9.5f} {('—' if zf is None else f'{zf:.3f}'):>7} "
          f"{time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"n4_{head}.npy", v)
g = np.corrcoef(RES["gen"], RES["q99"])[0, 1]
print(f"\ncorr(gen, q99) = {g:.5f}", flush=True)
print("порог допуска в стек: компонента полезна, если corr(L, новая) > corr(L, стек) * g")
print(f"  corr(L, gen) = {np.corrcoef(ly, RES['gen'])[0,1]:.5f}")
print(f"  corr(L, q99) = {np.corrcoef(ly, RES['q99'])[0,1]:.5f}")
print("\ndone")
