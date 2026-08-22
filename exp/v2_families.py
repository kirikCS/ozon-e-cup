"""Смена семейства на событийной оси: decoder-only и двусторонний энкодер с окном.

evgru2 и event_multi рекуррентны — состояние идёт слева направо и сжимает всё прошлое в один
вектор. Здесь два принципиально других считывания той же последовательности событий.

  base   — контроль, GRU;
  declm  — decoder-only: причинный трансформер над 96 токенами событий, как в TimesFM и Timer-XL.
           Событие t видит все предыдущие НАПРЯМУЮ, а не через сжатое состояние. Для клиента,
           у которого важна покупка полугодовой давности, это разные вещи;
  bienc  — двусторонний энкодер с ОГРАНИЧЕННЫМ окном: внимание разрешено только в полосе +-16
           событий, плюс токен CLS, который видит всё и виден всем. Двусторонность даёт событию
           контекст справа (что было ПОСЛЕ него), чего нет ни у GRU, ни у причинного декодера,
           а окно не даёт вниманию размазаться по всем 96 слотам.

Паддинг слева. У обеих рук нулевой токен (CLS/BOS) никогда не маскируется, и полоса всегда его
пропускает, — иначе у запроса, стоящего в паддинге, все ключи оказались бы закрыты и softmax
вернул бы NaN. Именно так ломается наивная связка attn_mask с key_padding_mask.

Пустой слот определяется по самому тензору: у настоящего события cos(2*pi*wd/7) не равен нулю
ни при каком дне недели.

Меряется и скор, и ортогональность: смена семейства должна двигать вектор до 0,988, тогда как
правка блока внутри семейства — только до 0,993-0,996.
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

FAM = "gru"

class TBlock(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * d, d))
    def forward(self, x, m, kpm):
        h = self.n1(x)
        x = x + self.att(h, h, h, attn_mask=m, key_padding_mask=kpm, need_weights=False)[0]
        return x + self.ff(self.n2(x))

class AttnTrunk(nn.Module):
    """причинный декодер либо двусторонний энкодер с полосой; токен 0 всегда жив"""
    def __init__(self, f=12, h=192, K=96, nl=3, band=16):
        super().__init__()
        self.kind, self.band, self.K = FAM, band, K
        self.inp = nn.Linear(f, h)
        self.cls = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, K + 1, h) * 0.02)
        self.bl = nn.ModuleList([TBlock(h) for _ in range(nl)])
        self.norm = nn.LayerNorm(h)
        self.out = 2 * h
        n = K + 1
        i = torch.arange(n).view(-1, 1)
        j = torch.arange(n).view(1, -1)
        m = (j > i) if FAM == "declm" else ((i - j).abs() > band)
        m[:, 0] = False                       # токен 0 виден всем: гарантия живого ключа
        m[0, :] = False
        self.register_buffer("mask", m)
    def forward(self, x):
        B = x.shape[0]
        pad = x.abs().sum(-1) == 0            # у события cos никогда не ноль
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        z = torch.cat([self.cls.expand(B, 1, -1), self.inp(x)], 1) + self.pos
        for b in self.bl:
            z = b(z, self.mask, kpm)
        z = self.norm(z)
        return torch.cat([z[:, 0], z[:, -1]], 1)   # CLS и самое свежее событие

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
        self.tr = nn.ModuleList([RNNTrunk(cell=cell) if FAM == "gru" else AttnTrunk(K=k)
                                 for k in ks])
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
for spec in os.environ.get("V2ARMS", "gru96:gru,gru96:declm,gru96:bienc").split(","):
    arm, famv = spec.split(":")
    globals()["FAM"] = famv
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
    np.save(OUT / f"v2_{arm}_{famv}.npy", v)
if len(RES) > 1:
    ks_ = list(RES)
    print("\nвзаимные корреляции рук:")
    print("             " + " ".join(f"{k:>12}" for k in ks_))
    for a_ in ks_:
        print(f"{a_:>12} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:12.5f}" for b in ks_))
    print("\nориентир: сид 0,999; архитектура 0,993-0,996; семейство 0,988")
    for nm in ("h8_gauss", "e2_gru96_sel", "q2c_daily_sel"):
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
