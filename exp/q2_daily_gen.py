"""Daily generative model: predict the 30-day path, not the 30-day total.

The scalar formulation gives the network 250k labels. The daily one gives it 250k x 30 =
7.5M. That density is the actual prize here, together with two things a scalar model cannot
express: WHEN inside the window a client buys, and what the calendar of those specific days
is — the test window contains Feb 23 and Mar 8.

The trap, and why this is not Tweedie again: summing predicted daily values estimates
E[Y|x], while RMSLE needs E[log1p Y|x], and log1p is concave. On a process where only 4.6%
of user-days carry a purchase the Jensen gap is enormous — this is exactly why raw Tweedie
(2.4401) and Poisson (2.4786) came in below the naive baseline. The fix is to make the model
GENERATIVE and take the expectation in the right place:

    y_hat = expm1( mean over sampled paths of log1p( sum_d y_d^path ) )

which is E[log1p Y|x] by construction, i.e. the RMSLE optimum itself.

Both estimators are computed and reported side by side, so the size of the trap is measured
rather than asserted.

Variants:
  indep   30 hazards and 30 value distributions emitted at once from the encoder state
  ar      autoregressive: a sampled purchase feeds back a "days since last purchase" signal,
          so the refractory period after a purchase is modelled
"""
import sys
import time
from datetime import date, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 364
CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
EPOCHS, BS, NPATH = 6, 1024, 256

print("строю тензор каналов ...")
chans = []
for n in CH_NAMES:
    try:
        P = panel(n).astype(np.float32)
    except Exception as e:
        print(f"  пропускаю {n}: {e}")
        continue
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"  SEQ {SEQ.shape}, каналов {CH}")

def cal_feats(anchor):
    """calendar covariates of the 30 horizon days of this anchor: dow one-hot + month cycle"""
    out = np.zeros((H, 10), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=anchor + 1 + i)
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

def daily_y(users, anchor):
    return G[users, anchor + 1:anchor + 1 + H].astype(np.float32)

class Enc(nn.Module):
    """dilated CNN only. The GRU read-out asked cuDNN for 24 GB of workspace on a 364-step
    sequence and was the sole cause of both OOMs; dilations to 64 already cover the window,
    which is exactly the geometry seq_pure uses without trouble. The recency the GRU was
    there for is recovered by pooling the last 30 days separately."""
    def __init__(self, c, h=128, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.h = h
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

class DailyHead(nn.Module):
    """per horizon day: hazard logit, mu and log-sigma of log1p(value | purchase)"""
    def __init__(self, hin, ncal=10, extra=0, d=128):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(hin + ncal + 16 + extra, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, h, cal, extra=None):
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                       cal.unsqueeze(0).expand(B, H, -1),
                       self.pos.unsqueeze(0).expand(B, H, -1)], -1)
        if extra is not None:
            z = torch.cat([z, extra], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.enc = Enc(c)
        self.head = DailyHead(self.enc.h * 3)
    def forward(self, x, cal):
        return self.head(self.enc(x), cal)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    bce = Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
    nll = 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)
    return (bce + pos * nll).sum(1).mean()

def fit(anchors, epochs=EPOCHS, seed=0):
    torch.manual_seed(seed)
    m = Model(CH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = epochs * len(anchors) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in anchors}
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        run, nb = 0.0, 0
        for a in anchors:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                yd = torch.from_numpy(daily_y(idx, a)).to(DEV)
                opt.zero_grad(set_to_none=True)
                l = loss_fn(*m(x, CAL[a]), yd)
                l.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
                run += l.item(); nb += 1
        print(f"    эпоха {ep}: loss {run/max(nb,1):.4f}", flush=True)
    return m.eval()

@torch.no_grad()
def predict(m, anchor, npath=NPATH, chunk=1024):
    """returns (MC estimate of E[log1p Y], naive log1p of E[Y]) — the trap, measured"""
    cal = torch.from_numpy(cal_feats(anchor)).to(DEV)
    mc = np.empty(NU, np.float32)
    naive = np.empty(NU, np.float32)
    for i in range(0, NU, chunk):
        idx = np.arange(i, min(i + chunk, NU))
        hz, mu, ls = m(torch.from_numpy(win(idx, anchor)).to(DEV), cal)
        p = torch.sigmoid(hz)
        sd = ls.exp()
        # наивная оценка: E[y_d] = p * E[expm1(N(mu,sd))] = p * (exp(mu + sd^2/2) - 1)
        ey = p * (torch.exp(mu + 0.5 * sd ** 2) - 1.0).clamp(min=0)
        naive[idx] = torch.log1p(ey.sum(1)).float().cpu().numpy()
        b = len(idx)
        acc = torch.zeros(b, device=DEV)
        for _ in range(npath // 64):
            u = torch.rand(64, b, H, device=DEV) < p.unsqueeze(0)
            v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0) * torch.randn(64, b, H, device=DEV))
            tot = (u.float() * v.clamp(min=0)).sum(-1)
            acc += torch.log1p(tot).sum(0)
        mc[idx] = (acc / (npath // 64 * 64)).float().cpu().numpy()
    return mc, naive

print(f"\n=== дневная генеративная модель, оценка на {ds(SEL_EV)} ===")
t0 = time.time()
m = fit(SEL_TR)
torch.cuda.empty_cache()
print(f"  обучение {time.time()-t0:.0f}s")
mc, naive = predict(m, SEL_EV)
y = target(SEL_EV)
print(f"\n  Монте-Карло E[log1p Y]  : RMSLE {rmsle(y, np.expm1(np.clip(mc,0,13))):.5f}")
print(f"  наивная log1p(E[Y])     : RMSLE {rmsle(y, np.expm1(np.clip(naive,0,13))):.5f}")
print(f"  величина ловушки Йенсена: {np.mean(naive-mc):.4f} в лог-шкале")

# диагностика предположения независимости дней
with torch.no_grad():
    cal = torch.from_numpy(cal_feats(SEL_EV)).to(DEV)
    idx = np.arange(0, 8192)
    hz, mu, ls = m(torch.from_numpy(win(idx, SEL_EV)).to(DEV), cal)
    p = torch.sigmoid(hz).cpu().numpy()
p0 = 1 - np.prod(1 - p, 1)
emp = (target(SEL_EV)[idx] > 0)
print(f"\n  диагностика: P(Y>0) модельная {p0.mean():.4f} против эмпирической {emp.mean():.4f}")
print(f"  (расхождение указывает на зависимость дней внутри окна)")
np.save("model_out/q2_daily_mc_sel.npy", mc)
print("\ndone")
