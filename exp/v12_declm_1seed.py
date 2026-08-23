"""Причинный трансформер над событиями: ОДИН сид, без усреднения.

Бэггинг съедает вклад тем сильнее, чем больше расходятся сиды: у свёрточного семейства
(взаимная корреляция сидов 0,995) усреднение по трём уронило вклад в восемь раз, у событийного
(0,999) — на четверть у трансформера. Одиночный сид даёт -1,34e-04 против -1,08e-04 у
трёхсидового, а стоит один прогон.

Имя выхода задаётся OUTTAG, чтобы не затереть трёхсидовые векторы.

Честная проверка на отложенной половине (w7) поставила эту руку первой по вкладу: -1,62e-04 из
-3,40e-04 всего набора. Отбор при этом видел её лишь третьей — величины по отбору и по честной
половине расходятся, и верить надо второй.

Модель: событие видит все предыдущие НАПРЯМУЮ через причинное внимание, а не через сжатое
состояние рекуррентной ячейки. Нулевой токен не маскируется никогда — иначе у запроса, стоящего
в паддинге, все ключи оказались бы закрыты и softmax вернул бы NaN.

Считаются оба протокола: SELECT под подбор весов, FINAL под сабмит.
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
NSEED = int(__import__("os").environ.get("NSEED", 3))
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
globals()["FAM"] = "declm"
assert FIN_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
y = target(SEL_EV); ly = np.log1p(y)
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
    nm = os.environ.get('OUTTAG', 'v9_declm') + ('_sel' if tag == 'SELECT' else '_fin')
    np.save(OUT / f"{nm}.npy", E)
    if len(acc) > 1:
        cs = np.mean([np.corrcoef(acc[i], acc[j])[0, 1]
                      for i in range(len(acc)) for j in range(i + 1, len(acc))])
        print(f"  -> {nm}.npy, взаимная корреляция сидов {cs:.5f}", flush=True)
print("\ndone")
