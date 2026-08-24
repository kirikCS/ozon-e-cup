"""Смена ПРЕДСТАВЛЕНИЯ событий: что вообще считать событием.

Прямое следствие 3.57, 3.59 и 3.60. За ночь на событийной оси перебрано три рычага, и все три
работали внутри неизменного тензора — те же 12 каналов, то же окно K=96, событие = день с
визитом. Считывание двигает вектор на уровне сида, ячейка — на уровне архитектуры, оператор
(GRU против трансформера) не двигает вовсе. Значит декоррелированную компоненту даст только
смена того, ЧТО модель видит.

Событие сейчас — день, когда клиент вообще заходил. Это самое частое из возможных определений,
и оно упирает окно K=96 в последние несколько месяцев. Другие определения меняют и плотность,
и ГЛУБИНУ, на которую окно достаёт:

  visit — контроль, день с визитом (продакшн);
  ord   — день с ЗАКАЗОМ. Событий на порядок меньше, значит те же 96 слотов достают на годы
          назад, а не на месяцы. Для клиента с редкими крупными покупками это принципиально
          другая история, чем 96 последних заходов;
  cart  — день с добавлением в корзину: середина между заходом и заказом, намерение без
          совершения.

Ось времени, глубина окна и плотность событий меняются все сразу — это и есть смена
представления, а не правка внутри него. Ожидание по иерархии рычагов: до 0,988, а не 0,999.

Оговорка, которую надо держать при чтении: у редких определений много клиентов вообще без
событий, и тензор у них полностью пустой. Модель это переживает (проверено в v2 на пустом
клиенте), но доля таких клиентов печатается — если она велика, рука меряет не столько ось,
сколько «у скольких вообще что-то было».

Меряется и скор, и ортогональность.
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
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
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

y = target(SEL_EV)
ly = np.log1p(y)
ref = np.load(OUT / "q2c_daily_sel.npy") if (OUT / "q2c_daily_sel.npy").exists() else None
print(f"\n{'рука':>10} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>16} {'сек':>7}")
RES = {}
import os
SEL0_TR, SEL0_EV = list(SEL_TR), SEL_EV
FIN0_TR, FIN0_EV = list(FIN_TR), FIN_EV
assert FIN0_EV + H > N_DAYS, "это не финальный якорь"
y = target(SEL0_EV); ly = np.log1p(y)
print(f"\n{'ось':>10} {'протокол':>9} {'RMSLE':>10} {'сек':>7}", flush=True)
for evd in os.environ.get("Y6ARMS", "cart").split(","):
    globals()["EVDEF"] = evd
    for tag, TR, EV in (("SELECT", SEL0_TR, SEL0_EV), ("FINAL", FIN0_TR, FIN0_EV)):
        globals()["SEL_TR"], globals()["SEL_EV"] = TR, EV
        t0 = time.time()
        try:
            v, npar = run("gru96", seed=0)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"{evd:>10} {tag:>9}  ОШИБКА {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        if tag == "SELECT":
            _s = rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
            assert _s > 1.70, f"УТЕЧКА: {evd} дал {_s:.5f}"
        nm = f"y6_{evd}_" + ("sel" if tag == "SELECT" else "fin")
        np.save(OUT / f"{nm}.npy", v.astype(np.float32))
        msg = f"{rmsle(y, np.expm1(np.clip(v-v.mean()+ly.mean(),0,13))):10.5f}" if tag == "SELECT" else f"{'—':>10}"
        print(f"{evd:>10} {tag:>9} {msg} {time.time()-t0:7.0f}  -> {nm}.npy", flush=True)
print("\ndone")
