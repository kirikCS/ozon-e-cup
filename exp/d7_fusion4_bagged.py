"""Production build with the FOURTH trunk: the event clock inside the fusion.

d5 settled where the event clock belongs. Standing alone it is useless — 1.75449 locally and
1.67191 on the leaderboard against a 1.66774 admission threshold, so its genuine orthogonality
(0.987 against 0.993-0.996 among calendar variants) is eaten by its own weakness. Inside the
fusion the same data helps: 1.73982 against 1.74013 for three trunks.

The asymmetry is the point. In a stack an orthogonal component must pay for its slot with its own
accuracy; inside a model it does not have to — the head simply reads one more view and weighs it
per client. That is why the event clock is folded in here rather than submitted separately.

Two results this rests on, both already measured.

d2: fusing a dilated CNN, attention over 14-day patches and attention over weekly tokens under
one generative head gives 1.74013 against 1.74117 for the CNN alone, and beats every single
trunk (best single: 1.74192). The trunks correlate 0.9926-0.9934 with each other — they read the
series differently — and the head combines rather than copies them.

d3: that gain is not capacity. A single trunk inflated to 1.91M parameters, MORE than fusion's
1.50M, scored 1.74380 — worse than the 0.42M original. Capacity actively hurts here; only the
diversity of readings helps.

Bagging is applied exactly as it was to the calendar model, where it bought -0.00222 locally and
-0.00252 on the leaderboard: four members, each trained on its own 80% subsample of clients, with
the frailty integrated by Gauss-Hermite quadrature rather than sampled. Members are averaged in
log space, which is where the metric lives.

Writes the SELECT vector (so the local gain is visible before a slot is spent) and the FINAL
vector at anchor 408.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, ds, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH = 364, 8, 512, 96, 16
PLEN, WLEN = 14, 7
NSEED, SUBSAMPLE, SIGMA = 4, 0.8, 0.8
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
EL = 2.3284
OUTCSV = "submissions/daily_fusion4_bagged.csv"

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

class CNNTrunk(nn.Module):
    def __init__(self, c, h=128, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.out = h * 3
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

class PatchTrunk(nn.Module):
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
        p = p.sum(-1).transpose(1, 2) if self.weekly else \
            p.permute(0, 2, 1, 3).reshape(B, self.n, CH * self.plen)
        z = self.emb(p) + self.pos
        for b in self.bl:
            z = b(z)
        return torch.cat([z.mean(1), z[:, -1]], -1)

ACT = (panel("visit") > 0)
NACT = 96

def win_active(users, end):
    """last NACT active days plus the log-gap before each"""
    out = np.zeros((len(users), CH + 1, NACT), np.float32)
    sub = ACT[users, :end + 1]
    for r, u in enumerate(users):
        idx = np.flatnonzero(sub[r])[-NACT:]
        if len(idx) == 0:
            continue
        out[r, :CH, -len(idx):] = SEQ[u][:, idx].astype(np.float32)
        out[r, CH, -len(idx):] = np.log1p(
            np.diff(np.concatenate([[idx[0]], idx])).astype(np.float32))
    return out

class ActTrunk(nn.Module):
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.out = h * 2
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z[:, :, -12:].mean(-1)], -1)

class Fusion(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.tr = nn.ModuleList([CNNTrunk(CH), PatchTrunk(CH, PLEN),
                                 PatchTrunk(CH, WLEN, weekly=True)])
        self.act = ActTrunk(CH + 1)
        tot = sum(t.out for t in self.tr) + self.act.out
        self.mix = nn.Sequential(nn.Linear(tot, 256), nn.SiLU(), nn.Linear(256, 192))
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(192 + 10 + 16, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal, xa):
        h = self.mix(torch.cat([t(x) for t in self.tr] + [self.act(xa)], -1))
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

def fit(anchors, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(NU, int(NU * SUBSAMPLE), replace=False))
    m = Fusion().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(anchors) * (len(keep) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in anchors}
    for _ in range(EPOCHS):
        m.train()
        for a in anchors:
            perm = keep[rng.permutation(len(keep))]
            for i in range(0, len(keep), BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                xa = torch.from_numpy(win_active(idx, a)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a], xa), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    return m.eval()

@torch.no_grad()
def simulate(m, anchor, chunk=256):
    cal = torch.from_numpy(cal_feats(anchor)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
    wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    for i in range(0, NU, chunk):
        idx = np.arange(i, min(i + chunk, NU))
        xa = torch.from_numpy(win_active(idx, anchor)).to(DEV)
        hz, mu, ls = m(torch.from_numpy(win(idx, anchor)).to(DEV), cal, xa)
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
    return out

for tag, TR, EV in (("SELECT", SEL_TR, SEL_EV), ("FINAL", FIN_TR, FIN_EV)):
    print(f"\n=== {tag}: {NSEED} членов по {SUBSAMPLE:.0%} клиентов ===", flush=True)
    acc = []
    for s in range(NSEED):
        t0 = time.time()
        m = fit(TR, seed=s)
        acc.append(simulate(m, EV))
        del m
        torch.cuda.empty_cache()
        msg = ""
        if tag == "SELECT":
            y = target(EV)
            ly = np.log1p(y)
            one = acc[-1] - acc[-1].mean() + ly.mean()
            ens = np.mean(acc, 0)
            ens = ens - ens.mean() + ly.mean()
            msg = (f"  член {rmsle(y, np.expm1(np.clip(one,0,13))):.5f}  "
                   f"среднее по {len(acc)}: {rmsle(y, np.expm1(np.clip(ens,0,13))):.5f}")
        print(f"  сид {s}:{msg}  ({time.time()-t0:.0f}s)", flush=True)
    E = np.mean(acc, 0)
    if tag == "SELECT":
        np.save(OUT / "d7_fusion4_sel.npy", E)
    else:
        f = E - E.mean() + EL
        pl.DataFrame({"user_id": np.load("cache/uids.npy"),
                      "predict": np.clip(np.expm1(f), 0, None)}).write_csv(OUTCSV)
        print(f"  {OUTCSV}, mean log1p {f.mean():.4f}")
print("\ndone")
