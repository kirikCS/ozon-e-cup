"""CoLES embeddings into the daily generative model — and the routing question, isolated.

Two facts make this the right pairing. CoLES, contrastive on the RAW daily sequences, probes at
1.73703 on its own and just entered the stack with the largest weight of any recovered component
(+0.0469) — it carries signal nothing else does. And the daily generative model sees no static
input at all: only the raw series and the horizon calendar. Giving it CoLES is adding information
it genuinely lacks.

g1 already ruled out the cheap version of this: contrastive over the 165 tabular COLUMNS probed
at 1.787-1.829, far worse than 1.737 on raw sequences. Compressing aggregates loses more than it
keeps. So the embedding here comes from the sequences, not the columns.

The arm that matters is the routing. I claimed that TFT-style static enrichment — embedding
through a GRN into a context vector that modulates the temporal path — beats plain concatenation,
because separating the static path is what keeps seq_pure at 0.9856 while seq_hybrid, which just
concatenates its tabular block, sits at 0.9944. That claim is testable here directly:

  base      daily generative model, no embedding            control
  concat    embedding appended to the head input            the naive route
  grn       embedding through a GRN into a context vector   the TFT route

If concat and grn tie, the routing claim is wrong and I should stop repeating it. If grn wins on
correlation while tying on score, the claim is about orthogonality, not accuracy — which is what
the stack actually pays for.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, OUT, CACHE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH, SIGMA = 364, 8, 512, 96, 16, 0.8
W_COLES, DIM, C_EPOCHS, C_BS, TEMP = 182, 96, 6, 512, 0.1
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

def win(users, end, length=L):
    s = end - length + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

def cal_feats(a):
    out = np.zeros((H, 10), np.float32)
    for i in range(H):
        d = D0 + timedelta(days=a + 1 + i)
        out[i, d.weekday()] = 1.0
        out[i, 7] = d.day / 31.0
        out[i, 8] = np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        out[i, 9] = np.cos(2 * np.pi * d.timetuple().tm_yday / 365.25)
    return out

# ---------------------------------------------------------------- CoLES
class ColesEnc(nn.Module):
    def __init__(self, c, h=128, dim=DIM, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.rnn = nn.GRU(h, h, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(h * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        r, _ = self.rnn(z.transpose(1, 2).contiguous())
        return Fn.normalize(self.proj(torch.cat([z.mean(-1), r[:, -1]], -1)), dim=-1)

def info_nce(a, b, temp=TEMP):
    lg = a @ b.T / temp
    t = torch.arange(len(a), device=a.device)
    return 0.5 * (Fn.cross_entropy(lg, t) + Fn.cross_entropy(lg.T, t))

def train_coles(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = ColesEnc(CH).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = C_EPOCHS * (NU // C_BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    for ep in range(C_EPOCHS):
        enc.train()
        perm = rng.permutation(NU)
        run = nb = 0
        for i in range(0, NU - C_BS, C_BS):
            u = np.sort(perm[i:i + C_BS])
            # два окна одного клиента, оба строго до последнего обучающего якоря — без утечки
            e1 = int(rng.integers(W_COLES // 2, SEL_TR[0] + 1))
            e2 = int(rng.integers(W_COLES // 2, SEL_TR[0] + 1))
            x1 = torch.from_numpy(win(u, e1, W_COLES)).to(DEV)
            x2 = torch.from_numpy(win(u, e2, W_COLES)).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = info_nce(enc(x1), enc(x2))
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step(); sch.step()
            run += loss.item(); nb += 1
        print(f"    CoLES эпоха {ep}: InfoNCE {run/max(nb,1):.4f}", flush=True)
    enc.eval()
    emb = {}
    with torch.no_grad():
        for a in SEL_TR + [SEL_EV]:
            out = np.empty((NU, DIM), np.float32)
            for i in range(0, NU, 8192):
                u = np.arange(i, min(i + 8192, NU))
                out[u] = enc(torch.from_numpy(win(u, a, W_COLES)).to(DEV)).cpu().numpy()
            emb[a] = out
    del enc
    torch.cuda.empty_cache()
    return emb

# ---------------------------------------------------------------- daily model
class GRN(nn.Module):
    def __init__(self, din, dh, dout, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(din, dh)
        self.fc2 = nn.Linear(dh, dout * 2)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)
        self.dp = nn.Dropout(drop)
    def forward(self, x):
        h = self.dp(self.fc2(Fn.elu(self.fc1(x))))
        a, b = h.chunk(2, -1)
        return self.norm(self.skip(x) + a * torch.sigmoid(b))

class Trunk(nn.Module):
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

class Model(nn.Module):
    def __init__(self, arm, d=128, dctx=64):
        super().__init__()
        self.arm = arm
        self.trunk = Trunk(CH)
        self.grn = GRN(DIM, 128, dctx) if arm == "grn" else None
        extra = DIM if arm == "concat" else (dctx if arm == "grn" else 0)
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(self.trunk.out + 10 + 16 + extra, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal, e=None):
        h = self.trunk(x)
        B = h.shape[0]
        parts = [h.unsqueeze(1).expand(B, H, -1),
                 cal.unsqueeze(0).expand(B, H, -1),
                 self.pos.unsqueeze(0).expand(B, H, -1)]
        if self.arm == "concat":
            parts.append(e.unsqueeze(1).expand(B, H, -1))
        elif self.arm == "grn":
            parts.append(self.grn(e).unsqueeze(1).expand(B, H, -1))
        o = self.net(torch.cat(parts, -1))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run(arm, EMB, seed=0):
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
                e = torch.from_numpy(EMB[a][idx]).to(DEV) if arm != "base" else None
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(x, CAL[a], e), yd).backward()
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
            e = torch.from_numpy(EMB[SEL_EV][idx]).to(DEV) if arm != "base" else None
            hz, mu, ls = m(torch.from_numpy(win(idx, SEL_EV)).to(DEV), cal, e)
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
    npar = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return out, npar

print("\n=== обучаю CoLES на сырых рядах (без меток) ===", flush=True)
t0 = time.time()
EMB = train_coles()
np.save(CACHE / "g4_coles_emb.npy", np.stack([EMB[a] for a in SEL_TR + [SEL_EV]]))
print(f"  {time.time()-t0:.0f}s", flush=True)

from sklearn.linear_model import Ridge
y = target(SEL_EV)
ly = np.log1p(y)
r = Ridge(alpha=10.0).fit(EMB[SEL_TR[0]], np.log1p(target(SEL_TR[0])))
print(f"  линейный зонд по эмбеддингам: "
      f"{rmsle(y, np.expm1(np.clip(r.predict(EMB[SEL_EV]),0,13))):.5f}  (у t21 было 1.73703)")

ref = OUT / "q2c_daily_sel.npy"
rv = np.load(ref) if ref.exists() else None
print(f"\n{'рука':>8} {'парам.':>9} {'RMSLE':>9} {'corr с дневной':>15} {'сек':>7}")
RES = {}
for arm in ("base", "concat", "grn"):
    t0 = time.time()
    try:
        v, npar = run(arm, EMB)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>8}  ОШИБКА {type(e).__name__}: {str(e)[:70]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly.mean()
    c = np.corrcoef(v, rv)[0, 1] if rv is not None else float("nan")
    print(f"{arm:>8} {npar/1e6:8.2f}М {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} "
          f"{c:15.5f} {time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"g4_{arm}_sel.npy", v)
if {"concat", "grn"} <= set(RES):
    def sc(k):
        v = RES[k]
        return rmsle(y, np.expm1(np.clip(v - v.mean() + ly.mean(), 0, 13)))
    print(f"\nвыигрыш эмбеддинга (concat − base): {sc('concat')-sc('base'):+.5f}")
    print(f"выигрыш эмбеддинга (grn − base):    {sc('grn')-sc('base'):+.5f}")
    print(f"маршрутизация (grn − concat):       {sc('grn')-sc('concat'):+.5f}")
    print(f"corr между concat и grn: {np.corrcoef(RES['concat'],RES['grn'])[0,1]:.5f}")
print("\ndone")
