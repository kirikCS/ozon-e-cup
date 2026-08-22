"""Самоулучшение энкодера s-типа: блок трансформера по деталям.

Геометрия энкодера v2 — 364 дня в 26 патчах по 14, токен CLS, обучаемый attention-пуллинг —
уже дала -0,00767 над старым усреднённым энкодером, и весь выигрыш пришёл из ВХОДА. Сам блок
с тех пор не трогали: пред-норма, обычное внимание с обучаемыми позициями, MLP с GELU.

  base   — контроль;
  rope   — вращательные позиции вместо обучаемых. Внимание начинает зависеть от РАЗНОСТИ
           позиций, а не от их абсолютных номеров. Для панели, у которой якоря обучения стоят
           в разных точках года, относительность — это ровно та инвариантность, которой не
           хватает: патч «14 дней назад» значит одно и то же при любом якоре;
  swiglu — вентильная MLP вместо GELU при том же числе параметров;
  alibi  — аддитивное смещение внимания -наклон*|i-j|, своё на голову. Мягкий приоритет
           близких патчей, который сеть может переучить, но не обязана выучивать с нуля;
  reg    — четыре регистровых токена: место, куда внимание может складывать глобальную сводку,
           не занимая под неё патч. В ViT это убирает артефакты внимания на случайных патчах.

Внимание здесь своё, а не nn.MultiheadAttention: RoPE требует доступа к q и k до скалярного
произведения, а модуль его не отдаёт. Рука base на своей реализации внимания повторяет
контрольное число — это проверяется первым делом.

Меряется и скор, и ортогональность.
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
for spec in os.environ.get("U3ARMS", "enc:base,enc:rope,enc:swiglu,enc:alibi,enc:reg").split(","):
    arm, varv = spec.split(":")
    globals()["VAR"] = varv
    t0 = time.time()
    v, npar = run(arm, SEL_TR, SEL_EV, EVT_S)
    np.save(OUT / f"u3_{arm}_{varv}.npy", v)
    CTRL[spec] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, rv)[0, 1] if rv is not None else float("nan")
    print(f"{spec:>12} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:15.5f} {time.time()-t0:7.0f}", flush=True)
ks_ = list(CTRL)
print("\nвзаимные корреляции рук:")
print("             " + " ".join(f"{k:>12}" for k in ks_))
for a_ in ks_:
    print(f"{a_:>12} " + " ".join(f"{np.corrcoef(CTRL[a_],CTRL[b])[0,1]:12.5f}" for b in ks_))
print("\nориентир: сид 0,999; архитектура 0,993-0,996; семейство 0,988")
for nm in ("h8_gauss", "f1_fusion_sel", "q2c_daily_sel"):
    f = OUT / f"{nm}.npy"
    if not f.exists():
        continue
    r = np.load(f)
    if r.shape != (NU,):
        continue
    print(f"\nкорреляция с {nm}:")
    for k in ks_:
        print(f"  {k:>12} {np.corrcoef(CTRL[k], r)[0,1]:.5f}")
print("\ndone")
