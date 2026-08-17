"""s-encoder, rebuilt — and the head separated from the trunk so we learn which one mattered.

The current s-encoder scores 1.6753-1.6785 on the leaderboard and sits inside `bwn` doing very
little. Three things differ from what works, and all three are in the wiring rather than in the
attention:

  window   182 days against 364 in the daily model. Long windows measurably carry signal — the 24
           long columns of the feature store alone score 1.79, and capping them costs 0.001.
  pooling  mean over 13 tokens, one statistic. The CNN pools mean+max, the GRU pools last+mean.
  head     a scalar: one logit and one mu for the whole 30-day window. No per-day resolution and
           no distribution — precisely the mechanism that put the daily model ahead of the trees.

Fixing all three at once would leave us unable to say which mattered, so the head is the arm:

  v2_scalar   new trunk (364 days, CLS token, attention pooling) + the OLD scalar head
  v2_gen      the same trunk + the generative head (30 hazards, 30 (mu, sigma), frailty,
              Gauss-Hermite quadrature, path simulation)
  v1_scalar   the current encoder, reproduced here for a like-for-like reference

v2_scalar - v1_scalar isolates the trunk. v2_gen - v2_scalar isolates the head. If the head
carries most of it, new encoders belong inside the fusion rather than as separate vectors, which
is what the event-clock result already suggested.

Attention pooling is a learned query over the token sequence rather than a mean, and a CLS token
gives the blocks somewhere to accumulate a global summary — two ways to the same end, both cheap,
so both are in and their concatenation is what the head reads.
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
EPOCHS, BS, NPATH, GH, SIGMA = 8, 512, 96, 16, 0.8
PLEN = 14
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
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

def win(users, end, L):
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

class Trunk(nn.Module):
    """v1: 182 days, mean pooling. v2: 364 days, CLS token plus learned attention pooling."""
    def __init__(self, c, L, d=96, nl=3, v2=False):
        super().__init__()
        self.L, self.v2, self.n = L, v2, L // PLEN
        self.emb = nn.Linear(c * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02) if v2 else None
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02) if v2 else None
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True) if v2 else None
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.out = d * 2 if v2 else d
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, PLEN).permute(0, 2, 1, 3).reshape(B, self.n, CH * PLEN)
        z = self.emb(p) + self.pos
        if self.v2:
            z = torch.cat([self.cls.expand(B, 1, -1), z], 1)
        for b in self.bl:
            z = b(z)
        if not self.v2:
            return z.mean(1)
        pooled = self.pool(self.q.expand(B, 1, -1), z[:, 1:], z[:, 1:], need_weights=False)[0]
        return torch.cat([z[:, 0], pooled.squeeze(1)], -1)

class ScalarHead(nn.Module):
    def __init__(self, hin, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hin, d), nn.BatchNorm1d(d), nn.SiLU(),
                                 nn.Linear(d, 2))
    def forward(self, h, cal=None):
        return self.net(h)

class GenHead(nn.Module):
    def __init__(self, hin, d=128):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(hin + 10 + 16, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, h, cal):
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                       cal.unsqueeze(0).expand(B, H, -1),
                       self.pos.unsqueeze(0).expand(B, H, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def scalar_loss(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def gen_loss(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(arm, seed=0):
    v2 = arm != "v1_scalar"
    gen = arm == "v2_gen"
    L = 364 if v2 else 182
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trunk = Trunk(CH, L, v2=v2).to(DEV)
    head = (GenHead(trunk.out) if gen else ScalarHead(trunk.out)).to(DEV)
    par = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(par, lr=1.5e-3, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in SEL_TR}
    for _ in range(EPOCHS):
        trunk.train(); head.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a, L)).to(DEV)
                h = trunk(x)
                opt.zero_grad(set_to_none=True)
                if gen:
                    yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                    gen_loss(*head(h, CAL[a]), yd).backward()
                else:
                    y = torch.from_numpy(target(a)[idx].astype(np.float32)).to(DEV)
                    scalar_loss(head(h), y).backward()
                nn.utils.clip_grad_norm_(par, 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    trunk.eval(); head.eval()
    cal = torch.from_numpy(cal_feats(SEL_EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    chunk = 256 if gen else 8192
    with torch.no_grad():
        for i in range(0, NU, chunk):
            idx = np.arange(i, min(i + chunk, NU))
            h = trunk(torch.from_numpy(win(idx, SEL_EV, L)).to(DEV))
            if not gen:
                o = head(h)
                out[idx] = (torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy()
                continue
            hz, mu, ls = head(h, cal)
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
    npar = sum(p.numel() for p in par)
    del trunk, head
    torch.cuda.empty_cache()
    return out, npar

y = target(SEL_EV)
ly = np.log1p(y)
ref = np.load(OUT / "q2c_daily_sel.npy") if (OUT / "q2c_daily_sel.npy").exists() else None
print(f"\n{'рука':>12} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>16} {'сек':>7}")
RES = {}
for arm in ("v1_scalar", "v2_scalar", "v2_gen"):
    t0 = time.time()
    try:
        v, npar = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>12}  ОШИБКА {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, ref)[0, 1] if ref is not None else float("nan")
    print(f"{arm:>12} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:16.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"e1_{arm}_sel.npy", v)
if "v1_scalar" in RES and "v2_scalar" in RES:
    a = rmsle(y, np.expm1(np.clip(RES["v1_scalar"] - RES["v1_scalar"].mean() + ly.mean(), 0, 13)))
    b = rmsle(y, np.expm1(np.clip(RES["v2_scalar"] - RES["v2_scalar"].mean() + ly.mean(), 0, 13)))
    print(f"\nвклад ствола (v2_scalar - v1_scalar): {b - a:+.5f}")
if "v2_scalar" in RES and "v2_gen" in RES:
    b = rmsle(y, np.expm1(np.clip(RES["v2_scalar"] - RES["v2_scalar"].mean() + ly.mean(), 0, 13)))
    c = rmsle(y, np.expm1(np.clip(RES["v2_gen"] - RES["v2_gen"].mean() + ly.mean(), 0, 13)))
    print(f"вклад головы (v2_gen - v2_scalar):   {c - b:+.5f}")
print("\ndone")
