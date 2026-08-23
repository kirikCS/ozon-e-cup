"""Углублённое окно в событийной модели стека: оба протокола, усреднение по сидам.

Единственная вещь из ночи, которая устояла после разбора утечки (3.68) и применима напрямую.
Это не новая компонента, а ПОЧИНКА существующей: продакшенный evgru2 работает при K=96 и обрезает
окно у 43,6% клиентов. При K=256 обрезается 4,3%, и замер даёт 1,74132 против 1,74850 — минус
0,00718. Больше, чем дала любая архитектурная правка за ночь: причинный трансформер −0,00245,
лучший пуллинг −0,00197. Рычаг «сколько модель видит» оказался сильнее всех рычагов «как она
смотрит».

Выигрыш здесь измерен ПРЯМО по скору, а не через стенд, поэтому разбор утечки его не задевает.

Считаются оба протокола:
  SELECT (якорь 348) — под подбор весов в стеке;
  FINAL  (якорь 408) — под сабмит. Цели за 408 не существует, поэтому скора нет, только вектор.

Усреднение по сидам обязательно: одиночная рука несёт заметную долю дисперсии от сида, а в стек
берётся направление, а не шум. У оси заказов взаимная корреляция сидов была 0,99973 — проверим и
здесь.

Батч и кусок инференса масштабированы под K: при K=256 память рекуррентного ствола втрое выше,
и зашитые под K=96 значения дважды роняли прогон по памяти видеокарты.
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
BS0, CHUNK0 = 1024, 8192
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
NSEED = int(__import__("os").environ.get("NSEED", 3))
WD0 = date(2025, 1, 1).weekday()

G, S = panel("gmv"), panel("srch")
CA, OR, V = panel("cart"), panel("ord"), panel("visit")
SC, SO = panel("scart"), panel("sord")
SD, CD = panel("sday"), panel("catday")

EVDEF, KWIN, CHUNK = "visit", 96, 8192

def build_events(a, K):
    """last K active days as a K x 12 event tensor, padded on the left"""
    src = {"visit": V, "ord": OR}[EVDEF]
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
    ks = [KWIN]
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
        for i in range(0, NU, CHUNK):
            idx = np.arange(i, min(i + CHUNK, NU))
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
globals()["EVDEF"], globals()["KWIN"] = "visit", 256
globals()["BS"] = max(64, BS0 * 96 // KWIN)
globals()["CHUNK"] = max(256, CHUNK0 * 96 // KWIN)
print(f"ось visit, окно K={KWIN}, батч {BS}, кусок {CHUNK}", flush=True)
assert FIN_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"

y = target(SEL_EV); ly = np.log1p(y)
ref = OUT / "v3_gru96_visit.npy"
rv = np.load(ref) if ref.exists() else None
if rv is not None:
    sh = rv - rv.mean() + ly.mean()
    print(f"продакшн K=96 (для сверки): {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}", flush=True)

for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    globals()["SEL_TR"], globals()["SEL_EV"] = TR, EV
    acc = []
    for sd in range(NSEED):
        t0 = time.time()
        v, npar = run("gru96", seed=sd)
        acc.append(v)
        E = np.mean(acc, 0)
        msg = ""
        if tag == "SELECT":
            sh = E - E.mean() + ly.mean()
            msg = f" среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}"
        print(f"  [{tag}] сид {sd}:{msg}  ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(acc, 0).astype(np.float32)
    nm = "v8_deep_sel" if tag == "SELECT" else "v8_deep_fin"
    np.save(OUT / f"{nm}.npy", E)
    if len(acc) > 1:
        cs = np.mean([np.corrcoef(acc[i], acc[j])[0, 1]
                      for i in range(len(acc)) for j in range(i + 1, len(acc))])
        print(f"  -> {nm}.npy, взаимная корреляция сидов {cs:.5f}", flush=True)
    if tag == "SELECT" and rv is not None:
        print(f"  corr с продакшенным K=96: {np.corrcoef(E, rv)[0,1]:.5f}")
        for n2 in ("h8_gauss", "v3_gru96_ord"):
            f2 = OUT / f"{n2}.npy"
            if f2.exists():
                print(f"  corr с {n2}: {np.corrcoef(E, np.load(f2))[0,1]:.5f}")
print("\ndone")
