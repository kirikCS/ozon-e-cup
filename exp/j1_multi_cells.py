"""Событийный ансамбль на LSTM вместо GRU.

Наблюдение владельца: в одиночном окне LSTM обошла GRU на 0.00084 (1.74718 против 1.74802,
эксперимент e2), а ансамбль трёх окон 32/96/182 собран на GRU. Если преимущество вентильного
устройства сохраняется при ансамблировании, там лежит непроверенный выигрыш.

Оговорка честная: 0.00084 всего вдвое выше полосы шума 0.0004, и одиночные GRU/LSTM стояли на
взаимной корреляции 0.9984 — то есть это одна модель с косметическим отличием. Ждать много не
стоит. Но event_multi_full СТОИТ в стеке с весом +0.149, а улучшение уже полезной компоненты
относится к шагам с S/N около единицы, в отличие от перекомбинаций весов с S/N 0.1-0.4.

Три руки: multi_lstm (все три ствола LSTM), multi_gru (контроль, повтор нынешнего рецепта тем же
кодом) и multi_mixed (GRU на коротком окне, LSTM на длинных) — последняя проверяет, не окажется
ли разнотипность стволов источником ортогональности, как это было с разнотипностью окон.

Полные данные, без подвыборки: h7 показал, что 80% ухудшают этот ансамбль монотонно.
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
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], 408
NSEED, SUBSAMPLE, EL = 4, 0.8, 2.3284
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
        if arm == "multi_lstm":
            cells = ["lstm"] * len(ks)
        elif arm == "multi_mixed":
            cells = ["gru"] + ["lstm"] * (len(ks) - 1)
        else:
            cells = ["gru"] * len(ks)
        self.tr = nn.ModuleList([RNNTrunk(cell=c) for c in cells])
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

def run(arm, seed=0, TR=None, EV=None, sub=1.0):
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    ks = [32, 96, 182] if arm.startswith("multi") else [96]
    # cuDNN резервирует буфер RNN на все шаги, все слои и все вентили сразу, поэтому он растёт
    # как batch x seqlen. В руке multi три ствола в одном проходе, длиннейший K=182, и на чанке
    # 8192 это запрос на 18 ГБ. Батч и чанк здесь урезаются пропорционально длине.
    bs = 512 if arm.startswith("multi") else BS
    chunk = 1024 if arm.startswith("multi") else 8192
    print(f"    строю событийные тензоры для K={ks} ...", flush=True)
    EVT = {(a, k): build_events(a, k) for a in TR + [EV] for k in ks}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Net(arm, ks).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    rng2 = np.random.default_rng(seed)
    keep = np.sort(rng2.choice(NU, int(NU*sub), replace=False)) if sub < 1 else np.arange(NU)
    steps = EPOCHS * len(TR) * (len(keep) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in TR}
    for _ in range(EPOCHS):
        m.train()
        for a in TR:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), bs):
                idx = np.sort(perm[i:i + bs])
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
        for i in range(0, NU, chunk):
            idx = np.arange(i, min(i + chunk, NU))
            xs = [torch.from_numpy(EVT[(EV, k)][idx].astype(np.float32)).to(DEV) for k in ks]
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
import polars as pl
NSEED_CMP = 2
y_ = target(SEL_EV); ly_ = np.log1p(y_)
print(f"\n=== сравнение вентилей: {NSEED_CMP} сида на полных данных, якорь {SEL_EV} ===", flush=True)
print(f"{'рука':>12} {'RMSLE':>9} {'сек':>7}", flush=True)
CMP = {}
for arm in ("multi_gru", "multi_lstm", "multi_mixed"):
    acc = []
    t0 = time.time()
    for s_ in range(NSEED_CMP):
        v, _ = run(arm, seed=s_, TR=SEL_TR, EV=SEL_EV, sub=1.0)
        acc.append(v)
    E = np.mean(acc, 0); E = E - E.mean() + ly_.mean()
    CMP[arm] = np.mean(acc, 0)
    print(f"{arm:>12} {rmsle(y_, np.expm1(np.clip(E,0,13))):9.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"j1_{arm}_sel.npy", np.mean(acc, 0))
print("\nвзаимные корреляции:")
ks_ = list(CMP)
print("             " + " ".join(f"{k:>12}" for k in ks_))
for a_ in ks_:
    print(f"{a_:>12} " + " ".join(f"{np.corrcoef(CMP[a_],CMP[b])[0,1]:12.4f}" for b in ks_))
best = min(CMP, key=lambda k: rmsle(y_, np.expm1(np.clip(CMP[k]-CMP[k].mean()+ly_.mean(),0,13))))
print(f"\nлучшая рука: {best}")
if best == "multi_gru":
    print("преимущество LSTM не перенеслось на ансамбль — финальный вектор не пересобираю")
else:
    print(f"\n=== FINAL: {best}, {NSEED} сидов на полных данных ===", flush=True)
    accf = []
    for s_ in range(NSEED):
        t0 = time.time()
        v, _ = run(best, seed=s_, TR=FIN_TR, EV=FIN_EV, sub=1.0)
        accf.append(v)
        print(f"  сид {s_} ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(accf, 0); f = E - E.mean() + EL
    pl.DataFrame({"user_id": np.load("cache/uids.npy"),
                  "predict": np.clip(np.expm1(f), 0, None)}).write_csv(f"submissions/event_{best}.csv")
    print(f"  submissions/event_{best}.csv, mean log1p {f.mean():.4f}")
print("\ndone")
