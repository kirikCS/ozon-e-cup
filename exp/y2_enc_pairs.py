"""Финальные версии энкодера s-типа: половины для разностей.

База считается один раз и служит старой половиной для rope, alibi, reg, swiglu. Энкодер — третья
ось после событийной (где измерен оператор, theta = -0,0149) и свёрточной, поэтому его разности
имеют шанс быть слабо зависимыми от уже измеренных.

RoPE и ALiBi дали одинаковый выигрыш по скору и совпадающую корреляцию — вероятно, это одно и то
же направление, и зондировать обе нет смысла. Считаю обе, но в зонд пойдёт одна, а вторая
послужит проверкой этой догадки через корреляцию разностей.
"""
import sys
import time
from datetime import date

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, PLEN = 364, 14
K_EV, H_EV = 96, 192
EPOCHS, BS, LR = 8, 512, 1.5e-3
NSEED, SUBSAMPLE = 4, 0.8
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
EL = 2.3284
WD0 = date(2025, 1, 1).weekday()

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G, S, CA, ORD_, V = (panel("gmv"), panel("srch"), panel("cart"), panel("ord"), panel("visit"))
SC, SO, SD, CD = panel("scart"), panel("sord"), panel("sday"), panel("catday")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def build_events(a):
    """last K_EV active days with explicit gaps — the evgru2 representation"""
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K_EV)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K_EV - cc[ui, jj]
    E = np.zeros((NU, K_EV, 12), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(ORD_[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    E[ui, slot, 8] = np.minimum(SC[ui, day], 20)
    E[ui, slot, 9] = np.minimum(SO[ui, day], 10)
    E[ui, slot, 10] = SD[ui, day]
    E[ui, slot, 11] = CD[ui, day]
    prev = np.full((NU, K_EV), -1, np.int64)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    valid = prev >= 0
    ds_ = np.roll(d0, 1, axis=1)
    vs = np.roll(valid, 1, axis=1)
    ds_[:, 0] = -1
    vs[:, 0] = False
    E[:, :, 4] = np.log1p(np.clip(np.where(valid & vs, d0 - ds_, 0), 0, None)).astype(np.float16)
    return E

VAR = "base"

def rope_tab(T, dh, device):
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    return torch.cos(torch.outer(torch.arange(T, device=device).float(), inv)), \
           torch.sin(torch.outer(torch.arange(T, device=device).float(), inv))

def apply_rope(x, tab):                      # x (B, nh, T, dh)
    cos, sin = tab
    c, s_ = cos.view(1, 1, x.shape[2], -1), sin.view(1, 1, x.shape[2], -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2)

def alibi_bias(nh, T, device):
    sl = torch.tensor([2.0 ** (-8.0 * (i + 1) / nh) for i in range(nh)], device=device)
    d = (torch.arange(T, device=device).view(-1, 1) - torch.arange(T, device=device).view(1, -1))
    return -sl.view(nh, 1, 1) * d.abs().float().view(1, T, T)

class MHA(nn.Module):
    """своё внимание: нужен доступ к q и k ради RoPE и аддитивного смещения"""
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.nh, self.dh, self.dp = nh, d // nh, drop
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
    def forward(self, x, bias=None, rope=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                            dropout_p=self.dp if self.training else 0.0)
        return self.o(o.transpose(1, 2).reshape(B, T, D))

class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.i = nn.Linear(d, 2 * mult * d)
        self.o = nn.Linear(mult * d, d)
    def forward(self, x):
        a, b = self.i(x).chunk(2, -1)
        return self.o(Fn.silu(a) * b)

class Block(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, nh, drop)
        self.ff = SwiGLU(d) if VAR == "swiglu" else nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
    def forward(self, x, bias=None, rope=None):
        x = x + self.att(self.n1(x), bias, rope)
        return x + self.ff(self.n2(x))

class EncTrunk(nn.Module):
    """геометрия v2: 364 дня в 26 патчах, CLS, обучаемый attention-пуллинг"""
    def __init__(self, c, d=96, nl=3, nh=4, nreg=4):
        super().__init__()
        self.var, self.nh = VAR, nh
        self.n = L // PLEN
        self.nreg = nreg if VAR == "reg" else 0
        self.emb = nn.Linear(c * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if self.nreg:
            self.reg = nn.Parameter(torch.randn(1, self.nreg, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, nh, batch_first=True)
        self.bl = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.out = d * 2
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, PLEN).permute(0, 2, 1, 3).reshape(B, self.n, CH * PLEN)
        e = self.emb(p)
        if self.var != "rope":
            e = e + self.pos
        parts = [self.cls.expand(B, 1, -1), e]
        if self.nreg:
            parts.append(self.reg.expand(B, self.nreg, -1))
        z = torch.cat(parts, 1)
        T = z.shape[1]
        rope = rope_tab(T, z.shape[-1] // self.nh, z.device) if self.var == "rope" else None
        bias = alibi_bias(self.nh, T, z.device).unsqueeze(0) if self.var == "alibi" else None
        for b in self.bl:
            z = b(z, bias, rope)
        pt = z[:, 1:1 + self.n]
        pooled = self.pool(self.q.expand(B, 1, -1), pt, pt, need_weights=False)[0]
        return torch.cat([z[:, 0], pooled.squeeze(1)], -1)

class EvTrunk(nn.Module):
    def __init__(self, f=12, h=H_EV, layers=2):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.rnn = nn.LSTM(h, h, num_layers=layers, batch_first=True, dropout=0.1)
        self.out = 2 * h
    def forward(self, x):
        o, _ = self.rnn(Fn.silu(self.inp(x)))
        return torch.cat([o[:, -1], o.mean(1)], 1)

class Net(nn.Module):
    def __init__(self, arm, d=160):
        super().__init__()
        self.enc = EncTrunk(CH) if arm in ("enc", "fusion") else None
        self.ev = EvTrunk() if arm in ("lstm", "fusion") else None
        tot = (self.enc.out if self.enc else 0) + (self.ev.out if self.ev else 0)
        self.body = nn.Sequential(nn.Linear(tot, d), nn.BatchNorm1d(d), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(d, 128), nn.BatchNorm1d(128),
                                  nn.SiLU())
        self.p, self.m = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, xs, xe):
        parts = []
        if self.enc is not None:
            parts.append(self.enc(xs))
        if self.ev is not None:
            parts.append(self.ev(xe))
        z = self.body(torch.cat(parts, -1))
        return torch.cat([self.p(z), self.m(z)], 1)

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def run(arm, TR, EV, EVT, seed=0, sub=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * sub), replace=False)) if sub < 1 else np.arange(NU)
    m = Net(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(TR) * (len(keep) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in TR}
    for _ in range(EPOCHS):
        m.train()
        for a in TR:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(win(idx, a)).to(DEV) if m.enc is not None else None
                xe = (torch.from_numpy(EVT[a][idx].astype(np.float32)).to(DEV)
                      if m.ev is not None else None)
                yb = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(m(xs, xe), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 2048):
            idx = np.arange(i, min(i + 2048, NU))
            xs = torch.from_numpy(win(idx, EV)).to(DEV) if m.enc is not None else None
            xe = (torch.from_numpy(EVT[EV][idx].astype(np.float32)).to(DEV)
                  if m.ev is not None else None)
            o = m(xs, xe)
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out), npar

print("строю событийные тензоры ...", flush=True)
t0 = time.time()
EVT_S = {a: build_events(a) for a in SEL_TR + [SEL_EV]}
print(f"  {time.time()-t0:.0f}s", flush=True)

y = target(SEL_EV)
ly = np.log1p(y)
ref = OUT / "q2c_daily_sel.npy"
rv = np.load(ref) if ref.exists() else None
print(f"\n{'рука':>9} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>15} {'сек':>7}")
CTRL = {}
import os
from common import H
assert FIN_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
y = target(SEL_EV); ly = np.log1p(y)
print(f"\n{'рука':>10} {'протокол':>9} {'RMSLE':>10} {'сек':>7}", flush=True)
EVT_F = {a: build_events(a) for a in FIN_TR + [FIN_EV]}
print(f"событийные тензоры: {len(EVT_S)} выборочных, {len(EVT_F)} финальных", flush=True)
for var in os.environ.get("Y2ARMS", "base,rope,alibi,reg").split(","):
    globals()["VAR"] = var
    for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
        t0 = time.time()
        try:
            v, npar = run("enc", TR, EV, EVT_S if tag == "SELECT" else EVT_F)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"{var:>10} {tag:>9}  ОШИБКА {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        nm = f"y2_{var}_" + ("sel" if tag == "SELECT" else "fin")
        np.save(OUT / f"{nm}.npy", v.astype(np.float32))
        msg = f"{rmsle(y, np.expm1(np.clip(v-v.mean()+ly.mean(),0,13))):10.5f}" if tag == "SELECT" else f"{'—':>10}"
        print(f"{var:>10} {tag:>9} {msg} {time.time()-t0:7.0f}  -> {nm}.npy", flush=True)
print("\ndone")
