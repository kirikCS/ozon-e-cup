"""Два победителя честной проверки вместе: причинный трансформер при ГЛУБОКОМ окне.

Честная проверка на отложенной половине оставила из ~40 рук два вектора, несущих почти всё:
  v2_gru96_declm  — причинный трансформер над событиями, -1,62e-04;
  v7_visit_256    — то же окно визитов, но 256 событий вместо 96, -1,14e-04.

Вместе они не пробовались. Причины ожидать выигрыша две, и они складываются:
глубина оказалась сильнейшим рычагом на этой оси (-0,00718 у рекуррентной модели), а трансформер
выигрывает у рекуррентности именно тем, что событие видит все предыдущие НАПРЯМУЮ, а не через
сжатое состояние, — и чем длиннее последовательность, тем эта разница должна быть весомее.
При K=96 рекуррентное состояние ещё справляется; при K=256 ему тяжелее, а вниманию всё равно.

Причина ожидать провала одна, и её надо назвать заранее: внимание квадратично по длине, 257
токенов против 97 — это в 7 раз больше пар. При батче 1024 матрицы внимания не поместятся, поэтому
батч уменьшен до 256. Уменьшение батча при том же числе эпох меняет число шагов оптимизатора, так
что рука отличается от v2_gru96_declm не только окном. Контрольная рука declm:96 считается ЗДЕСЬ
ЖЕ с тем же батчем — сравнивать надо с ней, а не с числом из v2.
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
EPOCHS, BS, LR = 7, 256, 1.2e-3
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

FAM, KWIN, CHUNK = "gru", 96, 8192

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
for spec in os.environ.get("V10ARMS", "declm:96,declm:256,gru:256").split(","):
    famv, kk = spec.split(":")
    arm = spec
    globals()["FAM"], globals()["KWIN"] = famv, int(kk)
    # внимание квадратично по длине: при K=256 кусок 8192 даёт матрицу на 8,7 ГБ.
    # на зашитом куске я уже падал дважды, поэтому он тоже масштабируется
    globals()["CHUNK"] = max(256, 8192 * 96 // int(kk))
    print(f"  [{spec}] семейство {famv}, окно K={KWIN}, батч {BS}, кусок {CHUNK}", flush=True)
    t0 = time.time()
    try:
        v, npar = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{spec:>16}  ОШИБКА {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    RES[spec] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, ref)[0, 1] if ref is not None else float("nan")
    print(f"{spec:>16} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:16.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"v10_{famv}_{kk}.npy", v)
if len(RES) > 1:
    ks_ = list(RES)
    print("\nвзаимные корреляции рук:")
    print("             " + " ".join(f"{k:>12}" for k in ks_))
    for a_ in ks_:
        print(f"{a_:>12} " + " ".join(f"{np.corrcoef(RES[a_],RES[b])[0,1]:12.5f}" for b in ks_))
    print("\nориентир: сид 0,999; архитектура 0,993-0,996; семейство 0,988")
    for nm in ("h8_gauss", "v2_gru96_declm", "v7_visit_256"):
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
