"""Пулинг событий: внимание с позиционным кодом против «последнее + среднее».

RNNTrunk сейчас считывает GRU как `cat([o[:,-1], o.mean(1)])` — самое свежее событие плюс
равновзвешенное среднее по всем. Среднее считает и левый паддинг: у клиента с 8 покупками
88 из 96 слотов — нули, и они разбавляют сигнал ровно в 12 раз.

Четыре руки разделяют три разных фактора, которые легко было бы смешать в один:
  gru96:base  — контроль, последнее + среднее;
  gru96:attn  — среднее ЗАМЕНЕНО на обучаемый запрос с позиционным кодом слота (без маски);
  gru96:attnm — то же плюс key_padding_mask на пустые слоты — отделяет вклад именно маски;
  gru96:cls   — последнее + среднее + запрос сверху, ДОБАВЛЕНИЕ поверх контроля.

Разделение attn/attnm существенно: в s1 я уже один раз подал маску вместе со сменой архитектуры
и не смог сказать, что из двух сработало. Здесь attn и attnm отличаются ровно маской.

Паддинг определяется из самого тензора: у настоящего события канал cos(2*pi*wd/7) не равен нулю
ни при каком дне недели, поэтому строка слота с нулевой суммой модулей — это гарантированно пустой
слот, а не событие с нулевой выручкой.

Позиционный код нужен именно пуллингу: сама GRU порядок уже знает, но запрос смотрит на выходы
как на МНОЖЕСТВО, и без кода не может спросить «покажи, что было в середине окна».

Меряется и скор, и ортогональность — вклад в стек идёт от смещения вектора, а не от скора руки.
"""
import sys
import time
from datetime import date

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 7, 1024, 1.2e-3
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
WD0 = date(2025, 1, 1).weekday()

G, S = panel("gmv"), panel("srch")
CA, OR, V = panel("cart"), panel("ord"), panel("visit")
SC, SO = panel("scart"), panel("sord")
SD, CD = panel("sday"), panel("catday")

def build_events(a, K):
    """last K active days as a K x 12 event tensor, padded on the left"""
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K - cc[ui, jj]
    E = np.zeros((NU, K, 12), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(OR[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    E[ui, slot, 8] = np.minimum(SC[ui, day], 20)
    E[ui, slot, 9] = np.minimum(SO[ui, day], 10)
    E[ui, slot, 10] = SD[ui, day]
    E[ui, slot, 11] = CD[ui, day]
    prev = np.full((NU, K), -1, np.int64)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    valid = prev >= 0
    ds_ = np.roll(d0, 1, axis=1); vs = np.roll(valid, 1, axis=1)
    ds_[:, 0] = -1; vs[:, 0] = False
    E[:, :, 4] = np.log1p(np.clip(np.where(valid & vs, d0 - ds_, 0), 0, None)).astype(np.float16)
    return E

POOL = "base"

class RNNTrunk(nn.Module):
    def __init__(self, f=12, h=192, cell="gru", layers=2, K=96):
        super().__init__()
        self.inp = nn.Linear(f, h)
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = R(h, h, num_layers=layers, batch_first=True, dropout=0.1)
        self.mode = POOL
        if POOL != "base":
            self.pe = nn.Parameter(torch.randn(1, K, h) * 0.02)
            self.q = nn.Parameter(torch.randn(1, 1, h) * 0.02)
            self.att = nn.MultiheadAttention(h, 4, batch_first=True)
        self.out = (3 if POOL == "cls" else 2) * h
    def forward(self, x):
        o, _ = self.rnn(Fn.silu(self.inp(x)))
        if self.mode == "base":
            return torch.cat([o[:, -1], o.mean(1)], 1)
        B = o.shape[0]
        kpm = None
        if self.mode == "attnm":
            kpm = x.abs().sum(-1) == 0        # пустой слот: у события cos никогда не ноль
            kpm = kpm.clone()
            kpm[:, -1] = False                # хотя бы один живой ключ, иначе NaN
        a = self.att(self.q.expand(B, 1, -1), o + self.pe, o + self.pe,
                     key_padding_mask=kpm, need_weights=False)[0].squeeze(1)
        if self.mode == "cls":
            return torch.cat([o[:, -1], o.mean(1), a], 1)
        return torch.cat([o[:, -1], a], 1)

class Net(nn.Module):
    def __init__(self, arm, ks):
        super().__init__()
        cell = "lstm" if arm == "lstm96" else "gru"
        self.tr = nn.ModuleList([RNNTrunk(cell=cell, K=k) for k in ks])
        tot = sum(t.out for t in self.tr)
        self.body = nn.Sequential(nn.Linear(tot, 160), nn.BatchNorm1d(160), nn.SiLU())
        self.p, self.m = nn.Linear(160, 1), nn.Linear(160, 1)
    def forward(self, xs):
        z = self.body(torch.cat([t(x) for t, x in zip(self.tr, xs)], -1))
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def run(arm, seed=0):
    ks = [32, 96, 182] if arm == "multi" else [96]
    print(f"    строю событийные тензоры для K={ks} ...", flush=True)
    EVT = {(a, k): build_events(a, k) for a in SEL_TR + [SEL_EV] for k in ks}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Net(arm, ks).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in SEL_TR}
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = [torch.from_numpy(EVT[(a, k)][idx].astype(np.float32)).to(DEV) for k in ks]
                yb = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(m(xs), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            xs = [torch.from_numpy(EVT[(SEL_EV, k)][idx].astype(np.float32)).to(DEV) for k in ks]
            o = m(xs)
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    npar = sum(p.numel() for p in m.parameters())
    del m, EVT
    torch.cuda.empty_cache()
    return np.concatenate(out), npar

y = target(SEL_EV)
ly = np.log1p(y)
ref = np.load(OUT / "q2c_daily_sel.npy") if (OUT / "q2c_daily_sel.npy").exists() else None
print(f"\n{'рука':>10} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>16} {'сек':>7}")
RES = {}
import os
ARMS = os.environ.get("EVARMS", "gru96:base,gru96:attn,gru96:attnm,gru96:cls").split(",")
for spec in ARMS:
    arm, pool = spec.split(":")
    globals()["POOL"] = pool
    t0 = time.time()
    try:
        v, npar = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>10}  ОШИБКА {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    RES[spec] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, ref)[0, 1] if ref is not None else float("nan")
    print(f"{arm:>10} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:16.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"s7_{arm}_{pool}.npy", v)
if len(RES) > 1:
    ks_ = list(RES)
    print("\nвзаимные корреляции рук (сколько пуллинг двигает вектор):")
    print("             " + " ".join(f"{k:>12}" for k in ks_))
    for a_ in ks_:
        print(f"{a_:>12} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:12.5f}" for b in ks_))
    print("\nориентир: смена сида даёт 0,999; смена семейства представления 0,988")
    for nm in ("h8_gauss", "e2_gru96_sel", "e2_multi_sel", "q2c_daily_sel"):
        f = OUT / f"{nm}.npy"
        if not f.exists():
            continue
        r = np.load(f)
        if r.shape != (NU,):
            continue
        print(f"\nкорреляция с {nm}:")
        for k in ks_:
            print(f"  {k:>12} {np.corrcoef(RES[k], r)[0,1]:.5f}")
print("\ndone")
