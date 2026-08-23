"""Многоосевая событийная модель: стволы на РАЗНЫХ определениях события.

v3 показал, что дефекты осей дополняют друг друга, и это прямо просится в конструкцию:

| ось | медиана событий | без событий вовсе | упирается в K=96 |
|---|---|---|---|
| визит | 82 | 0,00% | 43,6% |
| заказ | 8 | 15,54% | 1,1% |
| корзина | 23 | 3,04% | 9,1% |

Ось визитов не оставляет никого без входа, но у 43,6% клиентов обрезана окном и дальше 96
заходов не видит. Ось заказов достаёт до конца истории у 99%, но у 15,5% клиентов пуста
полностью, и модель выдаёт им константу — это механическая часть её отставания (+0,01596),
а не свойство покупочного времени.

Нынешний event_multi ставит три ствола на РАЗНЫЕ ДЛИНЫ ОКНА одной и той же оси визитов (32, 96,
182). Здесь три ствола стоят на РАЗНЫХ ОСЯХ при одной длине. Это другая конструкция: длины окна
дают одну и ту же информацию с разной детальностью, а оси — разную информацию.

  visit     — контроль, одна ось визитов (должен воспроизвести 1,74850);
  ord_visit — визиты и заказы: минимальная пара, где дефекты гасят друг друга;
  multi     — визиты, заказы, корзины.

Ожидание по ортогональности сдержанное и его стоит записать ДО замера: v3 дал 0,98293 для чистой
оси заказов, но добавление оси визитов возвращает в модель ровно ту информацию, на которой стоит
весь стек. По механизму, воспроизведённому трижды, скор вырастет, а корреляция вырастет вместе с
ним. Если так и выйдет — самой ортогональной останется одноосевая ord из v3, и брать надо её.
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

def build_events(a, K, evd="visit"):
    """last K active days as a K x 12 event tensor, padded on the left"""
    mask = {"visit": V, "ord": OR, "cart": CA}[evd][:, :a + 1] > 0
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

class RNNTrunk(nn.Module):
    def __init__(self, f=12, h=192, cell="gru", layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = R(h, h, num_layers=layers, batch_first=True, dropout=0.1)
        self.out = 2 * h
    def forward(self, x):
        o, _ = self.rnn(Fn.silu(self.inp(x)))
        return torch.cat([o[:, -1], o.mean(1)], 1)

class Net(nn.Module):
    def __init__(self, arm, ks):
        super().__init__()
        cell = "lstm" if arm == "lstm96" else "gru"
        self.tr = nn.ModuleList([RNNTrunk(cell=cell) for _ in ks])
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
    ks = {"visit": ["visit"], "ord_visit": ["visit", "ord"],
          "multi": ["visit", "ord", "cart"]}[arm]
    print(f"    строю событийные тензоры для осей {ks} при K=96 ...", flush=True)
    EVT = {(a, k): build_events(a, 96, k) for a in SEL_TR + [SEL_EV] for k in ks}
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
for arm in os.environ.get("V4ARMS", "visit,ord_visit,multi").split(","):
    t0 = time.time()
    try:
        v, npar = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>10}  ОШИБКА {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, ref)[0, 1] if ref is not None else float("nan")
    print(f"{arm:>10} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:16.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"v4_{arm}.npy", v)
if len(RES) > 1:
    ks_ = list(RES)
    print("\nвзаимные корреляции рук:")
    print("             " + " ".join(f"{k:>12}" for k in ks_))
    for a_ in ks_:
        print(f"{a_:>12} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:12.5f}" for b in ks_))
    print("\nориентир: сид 0,999; архитектура 0,993-0,996; смена представления 0,988")
    for nm in ("h8_gauss", "v3_gru96_ord", "e2_gru96_sel"):
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
