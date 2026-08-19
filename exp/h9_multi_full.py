"""Пересборка event_multi по ИСПРАВЛЕННОМУ рецепту: полные данные, усреднение сидов.

h7 закрыл вопрос, из-за чего провалилась первая сборка. Разброс по сидам на полных данных всего
0.00048 (1.74156 / 1.74204 / 1.74125), и усреднение там РАБОТАЕТ: 1.74156 -> 1.74147 -> 1.74122.
А подвыборка на 80% давала монотонное ухудшение 1.74186 -> 1.74331 -> 1.74332 -> 1.74367.

Значит виновата подвыборка, а не инициализация: три GRU на окнах 32/96/182 требуют полных
данных, при 80% каждый член деградирует согласованно, и усреднение этого не лечит.

Это не новая идея, а починка рецепта у единственной компоненты последней четвёрки, которую стек
оценил положительно: event_multi_bagged вошёл с весом +0.157 ДАЖЕ с испорченной сборкой, при
том что локально был худшим из четырёх. Собранный правильно, он должен быть заметно сильнее.
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

def run(arm, seed=0, TR=None, EV=None, sub=1.0):
    TR = SEL_TR if TR is None else TR
    EV = SEL_EV if EV is None else EV
    ks = [32, 96, 182] if arm == "multi" else [96]
    # cuDNN резервирует буфер RNN на все шаги, все слои и все вентили сразу, поэтому он растёт
    # как batch x seqlen. В руке multi три ствола в одном проходе, длиннейший K=182, и на чанке
    # 8192 это запрос на 18 ГБ. Батч и чанк здесь урезаются пропорционально длине.
    bs = 512 if arm == "multi" else BS
    chunk = 1024 if arm == "multi" else 8192
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
for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    print(f"\n=== {tag}: {NSEED} сидов на ПОЛНЫХ данных ===", flush=True)
    acc = []
    for s_ in range(NSEED):
        t0 = time.time()
        v, _ = run("multi", seed=s_, TR=TR, EV=EV, sub=1.0)   # полные данные, не 80%
        acc.append(v)
        msg = ""
        if tag == "SELECT":
            yv = target(EV); lyv = np.log1p(yv)
            e = np.mean(acc, 0); e = e - e.mean() + lyv.mean()
            msg = f"  среднее по {len(acc)}: {rmsle(yv, np.expm1(np.clip(e,0,13))):.5f}"
        print(f"  сид {s_}:{msg}  ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(acc, 0)
    if tag == "SELECT":
        np.save(OUT / "h9_multi_sel.npy", E)
    else:
        f = E - E.mean() + EL
        pl.DataFrame({"user_id": np.load("cache/uids.npy"),
                      "predict": np.clip(np.expm1(f), 0, None)}).write_csv("submissions/event_multi_full.csv")
        print(f"  submissions/event_multi_full.csv, mean log1p {f.mean():.4f}")
print("\ndone")
for _arm in ():
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
    np.save(OUT / f"e2_{arm}_sel.npy", v)
if len(RES) > 1:
    print("\nвзаимные корреляции:")
    ks = list(RES)
    print("        " + " ".join(f"{k:>9}" for k in ks))
    for a_ in ks:
        print(f"{a_:>8} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:9.4f}" for b in ks))
print("\ndone")
