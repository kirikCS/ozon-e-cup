"""The same decomposition, but the factors are learned jointly and the combination is trained
against the actual objective — which is what a GPU is for.

c1 fits each factor separately and then combines them. Every separate fit optimises its own
loss, and none of them optimises RMSLE on the total. A joint network can: one encoder, one head
per factor, and a final combination whose gradient flows back into all of them, so a factor is
pushed to be accurate exactly where accuracy in it changes the total.

Three combination rules, all sharing the same trunk and heads, differing only in how the pieces
are assembled — which isolates the combination rule as the variable under test:

  product   log1p(Y) = gate * (k + vpc)                the hard-coded rule c1 also tests
  gated     log1p(Y) = gate * (a*k + b*vpc + c)        the same rule with fitted coefficients:
                                                       if a and b drift far from 1, the product
                                                       was wrong in a measurable way
  learned   log1p(Y) = gate * MLP([k, vpc, conv, aov, h])   the combination as a small network
                                                       over the factors and the trunk state

The factor heads carry their own losses (BCE for the gate, masked L2 for the rest) alongside the
loss on the total, with a weight lambda. lambda = 0 makes the factors pure free parameters and
the model becomes an ordinary two-stage network — that is the control which says whether
factorising the intermediate quantities helps at all, or whether only the extra parameters do.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
EPOCHS, BS, LR = 8, 2048, 1.5e-3
G = panel("gmv")
ORD = panel("ord").astype(np.float32)
CART = panel("cart").astype(np.float32)

anchors = sorted(set(SEL_TR + [SEL_EV]))
X, names = build_features(anchors, verbose=False)
MU = np.concatenate([X[a] for a in SEL_TR], 0).mean(0)
SD = np.maximum(np.concatenate([X[a] for a in SEL_TR], 0).std(0), 1e-2)
Xs = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in anchors}
F = Xs[SEL_EV].shape[1]

def tgt(a):
    y = target(a).astype(np.float32)
    k = CART[:, a + 1:a + 31].sum(1)
    o = ORD[:, a + 1:a + 31].sum(1)
    return (y, np.log1p(k).astype(np.float32), np.log1p(o).astype(np.float32))

T = {a: tgt(a) for a in anchors}

class Net(nn.Module):
    def __init__(self, f, mode, d=256):
        super().__init__()
        self.mode = mode
        self.trunk = nn.Sequential(nn.Linear(f, d), nn.BatchNorm1d(d), nn.SiLU(), nn.Dropout(0.15),
                                   nn.Linear(d, d), nn.BatchNorm1d(d), nn.SiLU())
        self.gate = nn.Linear(d, 1)
        self.fac = nn.Linear(d, 4)          # k, vpc, conv, aov
        if mode == "gated":
            self.ab = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
        if mode == "learned":
            self.comb = nn.Sequential(nn.Linear(d + 4, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x):
        h = self.trunk(x)
        g = self.gate(h).squeeze(-1)
        f = self.fac(h)
        k, vpc = f[:, 0], f[:, 1]
        if self.mode == "product":
            m = k + vpc
        elif self.mode == "gated":
            m = self.ab[0] * k + self.ab[1] * vpc + self.ab[2]
        else:
            m = self.comb(torch.cat([f, h], 1)).squeeze(-1)
        return g, m, f

def run(mode, lam, seed=0):
    torch.manual_seed(seed)
    net = Net(F, mode).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        net.train()
        for a in SEL_TR:
            y, lk, lo = T[a]
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                xb = torch.from_numpy(Xs[a][idx]).to(DEV)
                yb = torch.from_numpy(y[idx]).to(DEV)
                kb = torch.from_numpy(lk[idx]).to(DEV)
                ob = torch.from_numpy(lo[idx]).to(DEV)
                g, m, f = net(xb)
                pos = (yb > 0).float()
                ly = torch.log1p(yb)
                loss = (Fn.binary_cross_entropy_with_logits(g, pos, reduction="none")
                        + pos * (m - ly) ** 2).mean()
                if lam > 0:
                    aux = (pos * ((f[:, 0] - kb) ** 2
                                  + (f[:, 1] - (ly - kb)) ** 2
                                  + (f[:, 2] - (ob - kb)) ** 2
                                  + (f[:, 3] - (ly - ob)) ** 2)).mean()
                    loss = loss + lam * aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, NU, 16384):
            idx = np.arange(i, min(i + 16384, NU))
            g, m, f = net(torch.from_numpy(Xs[SEL_EV][idx]).to(DEV))
            out.append((torch.sigmoid(g) * m.clamp(min=0)).cpu().numpy())
    ab = net.ab.detach().cpu().numpy() if mode == "gated" else None
    del net
    torch.cuda.empty_cache()
    return np.concatenate(out), ab

y = target(SEL_EV)
print(f"устройство {DEV}, оценка {ds(SEL_EV)}\n")
print(f"{'режим':>10} {'lambda':>7} {'RMSLE':>9} {'коэффициенты':>26}")
base = None
res = {}
for mode in ("product", "gated", "learned"):
    for lam in (0.0, 0.3):
        if mode == "product" and lam == 0.0:
            continue          # без вспомогательных лоссов факторы в product не определены
        t0 = time.time()
        p, ab = run(mode, lam)
        r = rmsle(y, np.expm1(np.clip(p, 0, 13)))
        res[f"{mode} lam={lam}"] = p
        if base is None:
            base = r
        extra = "" if ab is None else f"a={ab[0]:+.3f} b={ab[1]:+.3f} c={ab[2]:+.3f}"
        print(f"{mode:>10} {lam:7.1f} {r:9.5f} {extra:>26}   ({time.time()-t0:.0f}s)", flush=True)
print("\nвзаимные корреляции:")
ks = list(res)
for a_ in ks:
    print(f"  {a_:>16} " + " ".join(f"{np.corrcoef(res[a_],res[b])[0,1]:.4f}" for b in ks))
print("\nчитать так: если 'gated' даёт a и b далеко от 1, произведение было неверным правилом;")
print("если 'learned' бьёт оба, комбинацию действительно стоит учить, а не задавать")
print("done")
