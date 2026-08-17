"""Pretrain the encoder on a 7-day horizon, then fine-tune on the 30-day target.

The premise: a 7-day forecast is closer in time, so the client's state has less room to drift and
the task carries a cleaner gradient. Whatever the encoder learns there about "who this client is"
should transfer, while the head is task-specific and gets retrained.

This is NOT the multi-horizon arm that failed in d1. There, heads for 7/14/30/60 were trained
jointly and competed for trunk capacity — the 30-day head had to share. Here the horizons are
SEQUENTIAL: the short task shapes the trunk first and then steps aside entirely.

The control that decides whether the result is readable at all: pretraining spends extra epochs,
so a win could simply be more training. `scratch_long` gets the same total epoch budget on the
30-day task alone. Only `pre_ft` minus `scratch_long` is evidence for transfer.

  scratch       8 epochs on 30 days                      reference
  scratch_long  12 epochs on 30 days                     compute-matched control
  pre_ft        4 epochs on 7 days, then 8 on 30 days    transfer, whole model fine-tuned
  pre_frozen    same, but the trunk is frozen in phase 2 how much the pretrained trunk alone
                                                         carries, with only the head adapting

Trunk is the v2 geometry from e1 — 364 days, CLS token, learned attention pooling. Scalar head
throughout: it makes the two phases share a head shape, and it keeps every arm comparable on the
one axis under test.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, PLEN, BS, LR = 364, 14, 512, 1.5e-3
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
H_SHORT, H_LONG = 7, 30

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def tgt(a, hz):
    return G[:, a + 1:a + 1 + hz].sum(1).astype(np.float32)

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
    """364 days, CLS token, learned attention pooling — the v2 geometry"""
    def __init__(self, c, d=96, nl=3):
        super().__init__()
        self.n = L // PLEN
        self.emb = nn.Linear(c * PLEN, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.out = d * 2
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, PLEN).permute(0, 2, 1, 3).reshape(B, self.n, CH * PLEN)
        z = torch.cat([self.cls.expand(B, 1, -1), self.emb(p) + self.pos], 1)
        for b in self.bl:
            z = b(z)
        pooled = self.pool(self.q.expand(B, 1, -1), z[:, 1:], z[:, 1:], need_weights=False)[0]
        return torch.cat([z[:, 0], pooled.squeeze(1)], -1)

def head(hin, d=128):
    return nn.Sequential(nn.Linear(hin, d), nn.BatchNorm1d(d), nn.SiLU(), nn.Linear(d, 2))

def loss_fn(o, y):
    pos = (y > 0).float()
    return (Fn.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
            + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()

def phase(trunk, hd, hz, epochs, rng, freeze=False):
    par = list(hd.parameters()) + ([] if freeze else list(trunk.parameters()))
    opt = torch.optim.AdamW(par, lr=LR, weight_decay=1e-5)
    steps = epochs * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: tgt(a, hz) for a in SEL_TR}
    for _ in range(epochs):
        trunk.train(not freeze); hd.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                with torch.set_grad_enabled(not freeze):
                    h = trunk(x)
                if freeze:
                    h = h.detach()
                y = torch.from_numpy(Y[a][idx]).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(hd(h), y).backward()
                nn.utils.clip_grad_norm_(par, 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()

def run(arm, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trunk = Trunk(CH).to(DEV)
    if arm.startswith("pre"):
        hd_s = head(trunk.out).to(DEV)
        phase(trunk, hd_s, H_SHORT, 4, rng)          # фаза 1: недельная цель
        del hd_s
        p0 = float(np.mean([p.abs().mean().item() for p in trunk.parameters()]))
        print(f"    после предобучения средняя |вес| ствола {p0:.4f}", flush=True)
    hd = head(trunk.out).to(DEV)                      # голова всегда новая
    ep = 12 if arm == "scratch_long" else 8
    phase(trunk, hd, H_LONG, ep, rng, freeze=(arm == "pre_frozen"))
    trunk.eval(); hd.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            idx = np.arange(i, min(i + 8192, NU))
            o = hd(trunk(torch.from_numpy(win(idx, SEL_EV)).to(DEV)))
            out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    del trunk, hd
    torch.cuda.empty_cache()
    return np.concatenate(out)

y = target(SEL_EV)
ly = np.log1p(y)
sub = np.arange(0, NU, 7)
print(f"\nдоля покупающих: за {H_SHORT} дней {np.mean(tgt(SEL_EV,H_SHORT)[sub]>0):.4f}, "
      f"за {H_LONG} дней {np.mean(y[sub]>0):.4f}")
print(f"\n{'рука':>13} {'эпох на 30д':>12} {'RMSLE':>9} {'сек':>7}")
RES = {}
for arm in ("scratch", "scratch_long", "pre_ft", "pre_frozen"):
    t0 = time.time()
    try:
        v = run(arm)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>13}  ОШИБКА {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly.mean()
    print(f"{arm:>13} {12 if arm=='scratch_long' else 8:12d} "
          f"{rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"e3_{arm}_sel.npy", v)
def sc(k):
    v = RES[k]
    return rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
if "pre_ft" in RES and "scratch_long" in RES:
    print(f"\nперенос сверх равного бюджета (pre_ft - scratch_long): {sc('pre_ft')-sc('scratch_long'):+.5f}")
if "pre_ft" in RES and "scratch" in RES:
    print(f"наивная разность (pre_ft - scratch):                  {sc('pre_ft')-sc('scratch'):+.5f}")
if "pre_frozen" in RES and "pre_ft" in RES:
    print(f"цена заморозки ствола (pre_frozen - pre_ft):          {sc('pre_frozen')-sc('pre_ft'):+.5f}")
print("\nчитать так: если наивная разность положительна, а сверх бюджета — нет,")
print("работало не предобучение, а лишние шаги")
print("\ndone")
