"""Мультипликативная обусловленность против шага обучения: последняя невыключенная альтернатива.

u5 опроверг мой диагноз из 3.61. Я объяснял развал FiLM взаимодействием поклиентного множителя
с BatchNorm; при GroupNorm батчевых статистик нет вообще, механизм невозможен по построению —
а film_gn всё равно даёт 1,88894 против контроля 1,74369. Каждая починка улучшала
(2,12065 -> 2,01984 -> 1,96067 -> 1,88894) и ни одна не приблизилась.

Остаётся ровно одна невыключенная причина: ШАГ. LR унаследован от h3 и подбирался под аддитивную
модель. У промодулированной сети градиент табличного блока идёт сразу через все шесть блоков —
gamma умножает выход каждого, — и тот же шаг для неё может быть просто велик. Если дело в этом,
уменьшение шага должно двигать результат к контролю монотонно.

  concat     — контроль при штатном шаге;
  film_gn    — та же рука, что в u5, при штатном шаге (воспроизведение);
  film_gn_3  — шаг втрое меньше;
  film_gn_10 — шаг в десять раз меньше.

Если и при LR/10 остаётся провал — дело не в шаге, и мультипликативная обусловленность свёрточного
ствола табличным блоком закрыта как направление, с четырьмя выключенными объяснениями.
Если результат ползёт к контролю — закрыт был только мой способ её обучать.
"""
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
NSEED, SUBSAMPLE, EL_LEVEL = 4, 0.8, 2.3284
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1

def log16(n):
    return np.log1p(panel(n).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], 1)
CH = SEQ.shape[1]
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

ANCH = sorted(set(SEL_TR + FIN_TR + [SEL_EV, FIN_EV]))
X, names = build_features(ANCH, verbose=False)
FS = len(names)
MU = np.mean([X[a].mean(0) for a in ANCH], 0)
SD = np.maximum(np.mean([X[a].std(0) for a in ANCH], 0), 1e-2)
ST = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in ANCH}
del X
print(f"  {FS} колонок", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class GRN(nn.Module):
    def __init__(self, din, dh, dout, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(din, dh)
        self.fc2 = nn.Linear(dh, dout * 2)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)
        self.dp = nn.Dropout(drop)
    def forward(self, x):
        h = self.dp(self.fc2(Fn.elu(self.fc1(x))))
        a, b = h.chunk(2, -1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))

NORM, USEFILM = "bn", False
NB = 6

def mknorm(h):
    return nn.BatchNorm1d(h) if NORM == "bn" else nn.GroupNorm(8, h)

class FiLM(nn.Module):
    """скаляры -> свои gamma и beta для КАЖДОГО блока; gamma ограничена (0, 2)"""
    def __init__(self, fs, h, nb=NB, dctx=64):
        super().__init__()
        self.h, self.nb = h, nb
        self.ctx = GRN(fs, 256, dctx)
        self.gb = nn.Linear(dctx, 2 * h * nb)
        nn.init.zeros_(self.gb.weight)
        nn.init.zeros_(self.gb.bias)          # старт из тождественного отображения
    def forward(self, xt):
        p = self.gb(self.ctx(xt)).view(-1, self.nb, 2, self.h)
        return 1.0 + torch.tanh(p[:, :, 0]), p[:, :, 1]

class FiLMBlock(nn.Module):
    """свёртка -> нормировка -> МОДУЛЯЦИЯ -> активация: каноническое место FiLM"""
    def __init__(self, h, d):
        super().__init__()
        self.c = nn.Conv1d(h, h, 3, padding=d, dilation=d)
        self.n = mknorm(h)
    def forward(self, z, g=None, b=None):
        z = self.n(self.c(z))
        if g is not None:
            z = z * g.unsqueeze(-1) + b.unsqueeze(-1)
        return Fn.silu(z)

class CNNEnc(nn.Module):
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32), fs=0):
        super().__init__()
        assert len(dil) == NB
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([FiLMBlock(h, d) for d in dil])
        self.film = FiLM(fs, h) if USEFILM else None
        self.out = h * 2
    def forward(self, x, xt=None):
        z = self.inp(x)
        G = B = None
        if self.film is not None:
            G, B = self.film(xt)
        for i, b_ in enumerate(self.bl):
            z = z + (b_(z) if G is None else b_(z, G[:, i], B[:, i]))
        return torch.cat([z.mean(-1), z.max(-1).values], -1)

class Model(nn.Module):
    def __init__(self, arm, h=256, dctx=64, drop=0.15):
        super().__init__()
        self.arm = arm
        self.enc = CNNEnc(CH, fs=FS)
        self.body = nn.Sequential(nn.Linear(self.enc.out + FS, h), nn.BatchNorm1d(h),
                                  nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p, self.m = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, xs, xt):
        z = torch.cat([self.enc(xs, xt), xt], 1)   # прямой путь есть у ВСЕХ рук
        z = self.body(z)
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def run(arm, seed=0, TR=None, EV=None, sub=1.0):
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * sub), replace=False)) if sub < 1 else np.arange(NU)
    m = Model(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (len(keep) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in TR}
    for _ in range(EPOCHS):
        m.train()
        for a in TR:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(win(idx, a)).to(DEV)
                xt = torch.from_numpy(ST[a][idx]).to(DEV)
                yb = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(m(xs, xt), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            o = m(torch.from_numpy(win(idx, EV)).to(DEV),
                  torch.from_numpy(ST[EV][idx]).to(DEV))
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out), npar

y = target(SEL_EV)
ly = np.log1p(y)
import os
print(f"\n{'рука':>8} {'парам.':>10} {'RMSLE':>9} {'corr с дневной':>15} {'сек':>7}", flush=True)
ref = OUT / "q2c_daily_sel.npy"
rv = np.load(ref) if ref.exists() else None
RES = {}
CFG = {"concat": ("bn", False, 1.0), "film_gn": ("gn", True, 1.0),
       "film_gn_3": ("gn", True, 1 / 3), "film_gn_10": ("gn", True, 0.1)}
LR0 = LR
for arm in os.environ.get("U6ARMS", "concat,film_gn,film_gn_3,film_gn_10").split(","):
    globals()["NORM"], globals()["USEFILM"], _m = CFG[arm]
    globals()["LR"] = LR0 * _m
    print(f"  [{arm}] норма={NORM} модуляция={USEFILM} шаг={LR:.2e}", flush=True)
    t0 = time.time()
    try:
        v, npar = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>8}  ОШИБКА {type(e).__name__}: {str(e)[:70]}", flush=True)
        continue
    RES[arm] = v
    np.save(OUT / f"u6_{arm}.npy", v)
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, rv)[0, 1] if rv is not None else float("nan")
    print(f"{arm:>8} {npar/1e6:9.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:15.5f} {time.time()-t0:7.0f}", flush=True)
if len(RES) > 1:
    ks_ = list(RES)
    print("\nвзаимные корреляции рук:")
    print("         " + " ".join(f"{k:>9}" for k in ks_))
    for a_ in ks_:
        print(f"{a_:>8} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:9.5f}" for b in ks_))
    print("\nориентир: сид 0,999; архитектура 0,993-0,996; семейство 0,988")
    for nm in ("h8_gauss", "u4_concat", "q2c_daily_sel"):
        f = OUT / f"{nm}.npy"
        if not f.exists():
            continue
        r = np.load(f)
        if r.shape != (NU,):
            continue
        print(f"\nкорреляция с {nm}:")
        for k in ks_:
            print(f"  {k:>8} {np.corrcoef(RES[k], r)[0,1]:.5f}")
print("\ndone")
