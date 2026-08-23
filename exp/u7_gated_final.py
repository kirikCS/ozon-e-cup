"""Вентильная свёртка как компонента стека: оба протокола, усреднение по сидам.

Честная проверка на отложенной половине (w7) поставила эту руку третьей по вкладу: -4,97e-05.
Показательно, что по ОДИНОЧНОМУ скору она худшая из пяти правок блока (1,74426 против 1,74125
у базы), а ModernTCN, лучший по скору, не попал в набор вовсе. Третий за сутки случай, когда
локальный скор ведёт не туда.

Механизм: tanh(conv_f) * sigmoid(conv_g) вместо SiLU. Свёртка перестаёт быть поточечно-монотонной
и может ЗАКРЫТЬ канал на конкретном дне — на панели, где 95% дней пустые, это операция, которой
у неё не было.

Считаются оба протокола: SELECT под подбор весов, FINAL под сабмит.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, ds, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH = 364, 8, 512, 96, 16
PLEN, WLEN = 14, 7
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
NSEED = int(__import__("os").environ.get("NSEED", 3))
SIGMA = 0.8

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    P = panel(n).astype(np.float32)
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def cal_feats(a):
    out = np.zeros((H, 10), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, d.weekday()] = 1.0
        out[i, 7] = d.day / 31.0
        out[i, 8] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        out[i, 9] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
    return out

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class Block(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * d, d))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))

VARIANT = "base"

class Gated(nn.Module):
    """вентиль WaveNet: одна свёртка даёт содержимое, вторая — открыт ли канал"""
    def __init__(self, h, d):
        super().__init__()
        self.f = nn.Conv1d(h, h, 3, padding=d, dilation=d)
        self.g = nn.Conv1d(h, h, 3, padding=d, dilation=d)
        self.bn = nn.BatchNorm1d(h)
    def forward(self, z):
        return self.bn(torch.tanh(self.f(z)) * torch.sigmoid(self.g(z)))

class SEBlock(nn.Module):
    """растянутая свёртка плюс канальное перевзвешивание по глобальному контексту"""
    def __init__(self, h, d, r=8):
        super().__init__()
        self.c = nn.Conv1d(h, h, 3, padding=d, dilation=d)
        self.bn = nn.BatchNorm1d(h)
        self.se = nn.Sequential(nn.Linear(h, h // r), nn.SiLU(), nn.Linear(h // r, h), nn.Sigmoid())
    def forward(self, z):
        z = Fn.silu(self.bn(self.c(z)))
        return z * self.se(z.mean(-1)).unsqueeze(-1)

class BigK(nn.Module):
    """ModernTCN: поканальное большое ядро плюс поточечное смешивание каналов"""
    def __init__(self, h, k=51):
        super().__init__()
        self.dw = nn.Conv1d(h, h, k, padding=k // 2, groups=h)
        self.bn = nn.BatchNorm1d(h)
        self.pw = nn.Sequential(nn.Conv1d(h, 2 * h, 1), nn.SiLU(), nn.Conv1d(2 * h, h, 1))
    def forward(self, z):
        return self.pw(self.bn(self.dw(z)))

class CNNTrunk(nn.Module):
    def __init__(self, c, h=128, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.var = VARIANT
        self.inp = nn.Conv1d(c, h, 5, padding=2)
        if VARIANT == "gated":
            self.bl = nn.ModuleList([Gated(h, d) for d in dil])
        elif VARIANT == "se":
            self.bl = nn.ModuleList([SEBlock(h, d) for d in dil])
        elif VARIANT == "bigk":
            self.bl = nn.ModuleList([BigK(h) for _ in dil])
        else:
            self.bl = nn.ModuleList([nn.Sequential(
                nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU())
                for d in dil])
        self.out = h * 3 + (2 * c if VARIANT == "instnorm" else 0)
    def forward(self, x):
        st = None
        if self.var == "instnorm":
            mu = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True)
            st = torch.cat([mu.squeeze(-1), sd.squeeze(-1)], -1)
            x = (x - mu) / (sd + 1e-3)
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        o = torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)
        return o if st is None else torch.cat([o, st], -1)

class PatchTrunk(nn.Module):
    """attention over patches; plen=14 gives 26 tokens, plen=7 with weekly sums gives 52"""
    def __init__(self, c, plen, d=128, nl=3, weekly=False):
        super().__init__()
        self.plen, self.weekly, self.n = plen, weekly, L // plen
        self.emb = nn.Linear(c if weekly else c * plen, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.out = d * 2
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, self.plen)
        p = p.sum(-1) .transpose(1, 2) if self.weekly else \
            p.permute(0, 2, 1, 3).reshape(B, self.n, CH * self.plen)
        z = self.emb(p) + self.pos
        for b in self.bl:
            z = b(z)
        return torch.cat([z.mean(1), z[:, -1]], -1)

class Model(nn.Module):
    def __init__(self, arm, d=128):
        super().__init__()
        if arm == "cnn":
            self.tr = nn.ModuleList([CNNTrunk(CH)])
        elif arm == "senc":
            self.tr = nn.ModuleList([PatchTrunk(CH, PLEN)])
        elif arm == "wenc":
            self.tr = nn.ModuleList([PatchTrunk(CH, WLEN, weekly=True)])
        else:
            self.tr = nn.ModuleList([CNNTrunk(CH), PatchTrunk(CH, PLEN),
                                     PatchTrunk(CH, WLEN, weekly=True)])
        tot = sum(t.out for t in self.tr)
        self.mix = nn.Sequential(nn.Linear(tot, 256), nn.SiLU(), nn.Linear(256, 192)) \
            if len(self.tr) > 1 else nn.Identity()
        hin = 192 if len(self.tr) > 1 else tot
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(hin + 10 + 16, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal):
        h = self.mix(torch.cat([t(x) for t in self.tr], -1))
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                       cal.unsqueeze(0).expand(B, H, -1),
                       self.pos.unsqueeze(0).expand(B, H, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(arm, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Model(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in SEL_TR}
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a]), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    cal = torch.from_numpy(cal_feats(SEL_EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            hz, mu, ls = m(torch.from_numpy(win(idx, SEL_EV)).to(DEV), cal)
            hz = hz * scale
            sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NPATH, len(idx), H, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    n_par = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return out, n_par

y = target(SEL_EV)
ly = np.log1p(y)
ref = np.load(OUT / "q2c_daily_sel.npy") if (OUT / "q2c_daily_sel.npy").exists() else None
print(f"\n{'ствол':>8} {'параметров':>11} {'RMSLE':>9} {'corr с текущей':>16} {'сек':>7}")
if ref is not None:
    rs = ref - ref.mean() + ly.mean()
    print(f"{'текущая':>8} {'—':>11} {rmsle(y, np.expm1(np.clip(rs,0,13))):9.5f} "
          f"{1.0:16.5f} {0:7d}")
RES = {}
import os
globals()["VARIANT"] = "gated"
assert FIN_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
y = target(SEL_EV); ly = np.log1p(y)
for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    globals()["SEL_TR"], globals()["SEL_EV"] = TR, EV
    acc = []
    for sd in range(NSEED):
        t0 = time.time()
        v, npar = run("cnn", seed=sd)
        acc.append(v)
        E = np.mean(acc, 0)
        msg = ""
        if tag == "SELECT":
            sh = E - E.mean() + ly.mean()
            msg = f" среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}"
        print(f"  [{tag}] сид {sd}:{msg}  ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(acc, 0).astype(np.float32)
    nm = "u7_gated_sel" if tag == "SELECT" else "u7_gated_fin"
    np.save(OUT / f"{nm}.npy", E)
    if len(acc) > 1:
        cs = np.mean([np.corrcoef(acc[i], acc[j])[0, 1]
                      for i in range(len(acc)) for j in range(i + 1, len(acc))])
        print(f"  -> {nm}.npy, взаимная корреляция сидов {cs:.5f}", flush=True)
print("\ndone")
