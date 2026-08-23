"""Ось заказов на ФИНАЛЬНОМ якоре: вектор под отправку.

Стенд (w1, 71 вектор в базисе, 30 кандидатов) дал этой оси −1,76e−03 сверх перемешанного
контроля. Это на порядок больше любого другого ночного кандидата и втрое больше крупнейшего
вклада за весь проект — лестница порогов давала −5,32e−04. Обе половины согласны
(−1,84e−03 и −1,70e−03), контроль −5,6e−06, то есть ноль.

Механизм был предсказан до замера и подтвердился: во всём базисе НЕТ ни одного вектора,
индексирующего покупочное время. Все стоят на календарной оси или на оси визитов, где у 43,6%
клиентов окно K=96 обрезано и дальше 96 заходов не видит ничего. Ось заказов достаёт до конца
истории у 99% клиентов. Это новая информация, а не новый способ смотреть на старую — потому
вклад и на порядок больше, чем у любой правки считывателя, ячейки или оператора.

Цена: одиночный скор у неё ХУДШИЙ среди осей (1,76446 против 1,74850), потому что у 15,5%
клиентов заказов нет вовсе и модель выдаёт им константу. Ровно тот случай, ради которого стенд
и существует: по одиночному скору эту руку следовало бы выбросить.

Здесь обучение на FIN_TR с предсказанием на 408. Цели за 408 не существует (панель 409 дней),
поэтому скора нет — только вектор. Усреднение по нескольким сидам: у одиночной руки заметная
доля дисперсии от сида, а вклад в стек берётся от направления, не от шума.
"""
import sys
import time
from datetime import date

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 7, 1024, 1.2e-3
SEL_TR, SEL_EV = [378, 350, 322, 294, 266], N_DAYS - 1   # ФИНАЛЬНЫЙ протокол
NSEED = int(__import__("os").environ.get("NSEED", 3))
WD0 = date(2025, 1, 1).weekday()

G, S = panel("gmv"), panel("srch")
CA, OR, V = panel("cart"), panel("ord"), panel("visit")
SC, SO = panel("scart"), panel("sord")
SD, CD = panel("sday"), panel("catday")

EVDEF = "visit"

def build_events(a, K):
    """last K active days as a K x 12 event tensor, padded on the left"""
    src = {"visit": V, "ord": OR, "cart": CA}[EVDEF]
    mask = src[:, :a + 1] > 0
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

assert SEL_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
globals()["EVDEF"] = "ord"
acc = []
for sd in range(NSEED):
    t0 = time.time()
    v, npar = run("gru96", seed=sd)
    acc.append(v)
    E = np.mean(acc, 0)
    print(f"  сид {sd}: усреднено {len(acc)}, sd вектора {E.std():.5f}, "
          f"среднее log1p {E.mean():.5f}  ({time.time()-t0:.0f}s)", flush=True)
E = np.mean(acc, 0).astype(np.float32)
np.save(OUT / "v6_ord_fin.npy", E)
print(f"\n-> v6_ord_fin.npy, параметров {npar/1e6:.2f}М")
print(f"взаимные корреляции сидов: "
      f"{np.mean([np.corrcoef(acc[i],acc[j])[0,1] for i in range(len(acc)) for j in range(i+1,len(acc))]):.5f}"
      if len(acc) > 1 else "")
print("\ndone")
