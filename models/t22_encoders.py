"""Single-tower and two-tower encoders over the daily panel, with attention variants.

Five configurations, differing along two axes:

  attention   softmax (standard scaled dot-product)
              linear  (kernel feature map, phi(q)(phi(k)'v) — O(n) instead of O(n^2))
              gated   (sigmoid gate on the value path, no normalisation across positions)

  fusion      s-encoder    sequence tower only
              concat       sequence + tabular, concatenated at the head (what seq_hybrid does)
              cross        tabular vector queries the sequence patches by cross-attention
              gated-bi     bilinear gate between the two towers

The point is not that any single one beats boosting — it is that they place their errors somewhere
new. The stack has been paying for orthogonality, not for accuracy, for several rounds now.
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "models")
from zoo_common import load, emit, TR_PROBE, TR_FINAL, FIT, PRED, EL
from common import N_DAYS, NU, panel, target, rmsle, ds, CACHE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L_DAY, PLEN = 182, 14
X, Y, NAMES, TOP = load()
FDIM = X[PRED].shape[1]

SEQ = np.stack([np.log1p(panel("gmv")).astype(np.float16),
                np.log1p(panel("gmv_search").astype(np.float32)).astype(np.float16),
                np.log1p(panel("ord").astype(np.float32)).astype(np.float16),
                np.log1p(panel("cart").astype(np.float32)).astype(np.float16),
                np.log1p(panel("srch").astype(np.float32)).astype(np.float16),
                panel("visit").astype(np.float16),
                panel("engaged").astype(np.float16)], 1)
CH = SEQ.shape[1]

def win(users, end):
    s = end - L_DAY + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

# ------------------------------------------------------------------ attention
class Attn(nn.Module):
    def __init__(self, d, nh=4, kind="softmax", drop=0.1):
        super().__init__()
        self.nh, self.d, self.kind = nh, d, kind
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.g = nn.Linear(d, d) if kind == "gated" else None
        self.dp = nn.Dropout(drop)
    def forward(self, q_in, kv):
        B, Tq, D = q_in.shape
        Tk = kv.shape[1]
        h = self.nh
        q = self.q(q_in).view(B, Tq, h, D // h).transpose(1, 2)
        k = self.k(kv).view(B, Tk, h, D // h).transpose(1, 2)
        v = self.v(kv).view(B, Tk, h, D // h).transpose(1, 2)
        if self.kind == "softmax":
            a = (q @ k.transpose(-1, -2)) / (D // h) ** 0.5
            z = self.dp(a.softmax(-1)) @ v
        elif self.kind == "linear":
            # phi(x) = elu(x)+1 keeps the kernel positive, then (phi(q) (phi(k)' v)) is O(T)
            qp, kp = Fn.elu(q) + 1, Fn.elu(k) + 1
            kv_ = kp.transpose(-1, -2) @ v
            z = (qp @ kv_) / (qp @ kp.sum(-2, keepdim=True).transpose(-1, -2) + 1e-6)
        else:                                    # gated: no softmax, a learned per-channel gate
            a = (q @ k.transpose(-1, -2)) / (D // h) ** 0.5
            z = torch.sigmoid(a) @ v / Tk
        z = z.transpose(1, 2).reshape(B, Tq, D)
        return self.o(z) * (torch.sigmoid(self.g(q_in)) if self.g is not None else 1.0)

class Block(nn.Module):
    def __init__(self, d, nh, kind, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attn(d, nh, kind, drop)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 2, d))
    def forward(self, x, kv=None):
        h = self.n1(x)
        x = x + self.att(h, h if kv is None else kv)
        return x + self.ff(self.n2(x))

class SeqTower(nn.Module):
    """patch embedding over the daily channels, then attention blocks"""
    def __init__(self, c, d=96, nl=3, nh=4, kind="softmax"):
        super().__init__()
        self.emb = nn.Linear(c * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, L_DAY // PLEN, d) * 0.02)
        self.bl = nn.ModuleList([Block(d, nh, kind) for _ in range(nl)])
        self.d = d
    def forward(self, x):
        B = x.shape[0]
        n = L_DAY // PLEN
        p = x.reshape(B, CH, n, PLEN).permute(0, 2, 1, 3).reshape(B, n, CH * PLEN)
        z = self.emb(p) + self.pos
        for b in self.bl:
            z = b(z)
        return z

class Model(nn.Module):
    def __init__(self, fdim, kind="softmax", fusion="concat", d=96, extra=0):
        super().__init__()
        self.fusion = fusion
        self.seq = SeqTower(CH, d, kind=kind)
        self.tab = nn.Sequential(nn.Linear(fdim + extra, 256), nn.BatchNorm1d(256), nn.SiLU(),
                                 nn.Dropout(0.15), nn.Linear(256, d), nn.BatchNorm1d(d), nn.SiLU())
        if fusion == "cross":
            self.cross = Block(d, 4, kind)
        if fusion == "gated-bi":
            self.gate = nn.Bilinear(d, d, d)
        din = d if fusion == "s" else (2 * d if fusion in ("concat", "cross") else d)
        self.head = nn.Sequential(nn.Linear(din, 128), nn.BatchNorm1d(128), nn.SiLU(),
                                  nn.Linear(128, 2))
    def forward(self, xs, xt):
        zs = self.seq(xs)
        if self.fusion == "s":
            z = zs.mean(1)
        else:
            zt = self.tab(xt)
            if self.fusion == "concat":
                z = torch.cat([zs.mean(1), zt], 1)
            elif self.fusion == "cross":
                q = self.cross(zt.unsqueeze(1), zs).squeeze(1)
                z = torch.cat([q, zt], 1)
            else:
                z = self.gate(zs.mean(1), zt)
        return self.head(z)

def loss_fn(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()
pred_fn = lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)

# optional CoLES embeddings as extra tabular columns
EMB = None
try:
    arr = np.load(CACHE / "coles_emb.npy")
    ANCH = sorted(set(TR_PROBE + TR_FINAL + [FIT, PRED]))
    EMB = {a: arr[i] for i, a in enumerate(ANCH)}
    print(f"CoLES-эмбеддинги подхвачены: {arr.shape[-1]} измерений")
except Exception as e:
    print("CoLES-эмбеддингов нет:", e)

def tabmat(a, use_emb):
    return np.concatenate([X[a], EMB[a]], 1) if (use_emb and EMB) else X[a]

def fit(kind, fusion, anchors, ev, use_emb=False, epochs=5, bs=2048, lr=1.5e-3):
    extra = EMB[ev].shape[1] if (use_emb and EMB) else 0
    torch.manual_seed(0)
    model = Model(FDIM, kind, fusion, extra=extra).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * len(anchors) * (NU // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10))
    TB = {a: tabmat(a, use_emb) for a in anchors}
    YT = {a: Y[a].astype(np.float32) for a in anchors}
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        model.train()
        for a in anchors:
            perm = rng.permutation(NU)
            for i in range(0, NU, bs):
                idx = np.sort(perm[i:i + bs])
                if len(idx) < 64: continue
                xs = torch.from_numpy(win(idx, a)).to(DEV)
                xt = torch.from_numpy(TB[a][idx]).to(DEV)
                yb = torch.from_numpy(YT[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xs, xt), yb)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    model.eval()
    TE = tabmat(ev, use_emb)
    acc = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            acc.append(pred_fn(model(torch.from_numpy(win(idx, ev)).to(DEV),
                                     torch.from_numpy(TE[idx]).to(DEV))).float().cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(acc).numpy()

JOBS = [
    ("s-encoder, softmax", "enc_s_softmax", "softmax", "s", False),
    ("s-encoder, linear attention", "enc_s_linear", "linear", "s", False),
    ("bi-encoder, cross-attention", "enc_bi_cross", "softmax", "cross", False),
    ("bi-encoder, gated bilinear", "enc_bi_gated", "gated", "gated-bi", False),
    ("bi-encoder, cross + CoLES", "enc_bi_cross_coles", "softmax", "cross", True),
]
print(f"=== энкодеры: {len(JOBS)} вариаций ===")
for name, fname, kind, fusion, ue in JOBS:
    print(f"\n{name}")
    t0 = time.time()
    try:
        lp = fit(kind, fusion, TR_PROBE, FIT, ue)
        print(f"    probe {ds(FIT)}: {rmsle(Y[FIT], np.expm1(np.clip(np.nan_to_num(lp),0,13))):.5f}"
              f"  ({time.time()-t0:.0f}s)")
        emit(fit(kind, fusion, TR_FINAL, PRED, ue), fname)
    except Exception as e:
        import traceback; print(f"    FAILED: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\ndone")
