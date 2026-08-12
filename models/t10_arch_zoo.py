"""Remaining independent architectures at anchor 2026-02-13.

Two sequence encoders (GRU over weekly aggregates, PatchTST-style transformer over daily patches)
and two tabular networks (wide MLP, MDME-style multi-expert head). The CNN hybrid took the largest
weight in the six-vector stack, so architectures that differ from both the trees and from that CNN
are the remaining source of orthogonal signal.

Each model: 1-seed probe at the validation anchor as a smoke test, then 3 seeds averaged in log
space for the submission, level-matched to the E[log1p(y)] already solved from the leaderboard.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from common import H, NU, N_DAYS, panel, target, rmsle, Timer, build_features, OUT, CACHE, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRED, FIT = N_DAYS - 1, 378
TR_TAB_F, TR_TAB_P = [378, 350, 322, 294, 266], [348, 320, 292, 264, 236]
TR_SEQ_F, TR_SEQ_P = [378, 350, 322, 294], [348, 320, 292, 264]
ALL_A = sorted(set(TR_TAB_F + TR_TAB_P + TR_SEQ_F + TR_SEQ_P + [PRED, FIT]))
L_DAY, W, SEEDS, EL = 182, 7, 3, 2.3284
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)
print("device:", DEV)

chans = [np.log1p(panel("gmv")).astype(np.float16),
         np.log1p(panel("gmv_search").astype(np.float32)).astype(np.float16),
         np.log1p(panel("ord").astype(np.float32)).astype(np.float16),
         np.log1p(panel("cart").astype(np.float32)).astype(np.float16),
         np.log1p(panel("srch").astype(np.float32)).astype(np.float16),
         panel("visit").astype(np.float16), panel("engaged").astype(np.float16)]
SEQ = np.stack(chans, 1); C = SEQ.shape[1]
X, NAMES = build_features(ALL_A)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
for a in ALL_A:
    X[a] = np.concatenate([X[a], BT[a]], 1)
FDIM = X[PRED].shape[1]
Y = {a: target(a) for a in ALL_A if a != PRED}
print(f"SEQ {SEQ.shape}, features {FDIM}")

def norm_stats(anchors):
    Z = np.nan_to_num(np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in anchors], 0),
                      posinf=0, neginf=0)
    return Z.mean(0), np.maximum(Z.std(0), 1e-2)

def tab(a, mu, sd):
    z = np.nan_to_num(np.log1p(np.clip(X[a], 0, None)), posinf=0, neginf=0)
    return np.clip((z - mu) / sd, -8, 8).astype(np.float32)

def seq_slice(a, idx, weekly):
    s = a - L_DAY + 1
    if s >= 0:
        win = SEQ[idx, :, s:a + 1].astype(np.float32)
    else:
        got = SEQ[idx, :, 0:a + 1].astype(np.float32)
        win = np.concatenate([np.zeros((len(idx), C, -s), np.float32), got], 2)
    return win.reshape(win.shape[0], C, L_DAY // W, W).sum(-1) if weekly else win

# ------------------------------------------------------------------ encoders
class GRUEnc(nn.Module):
    def __init__(self, c, h=128):
        super().__init__(); self.rnn = nn.GRU(c, h, 2, batch_first=True, dropout=0.1); self.out = h * 2
    def forward(self, x):
        z, _ = self.rnn(x.transpose(1, 2))
        return torch.cat([z[:, -1], z.mean(1)], -1)

class PatchEnc(nn.Module):
    def __init__(self, c, plen=14, d=128, nl=3, nh=4):
        super().__init__()
        self.plen = plen; self.emb = nn.Linear(c * plen, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, nh, d * 4, dropout=0.1, batch_first=True,
                                        norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(el, nl); self.out = d * 2
    def forward(self, x):
        B, c, T = x.shape; n = T // self.plen
        p = x[:, :, :n * self.plen].reshape(B, c, n, self.plen).permute(0, 2, 1, 3).reshape(B, n, c * self.plen)
        z = self.tr(self.emb(p) + self.pos[:, :n])
        return torch.cat([z.mean(1), z[:, -1]], -1)

class SeqHybrid(nn.Module):
    def __init__(self, enc, d_tab, h=256, drop=0.15):
        super().__init__()
        self.enc = enc; d = enc.out + d_tab
        self.body = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p = nn.Linear(128, 1); self.m = nn.Linear(128, 1)
    def forward(self, xs, xt):
        z = self.body(torch.cat([self.enc(xs), xt], 1))
        return torch.cat([self.p(z), self.m(z)], 1)

class MLP(nn.Module):
    def __init__(self, f, h=(1024, 512, 256), out=1, drop=0.15):
        super().__init__()
        L, d = [], f
        for k in h:
            L += [nn.Linear(d, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(drop)]; d = k
        L += [nn.Linear(d, out)]; self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)

class MDME(nn.Module):
    def __init__(self, f, n_b=8, h=256, drop=0.15):
        super().__init__()
        self.n_b = n_b
        self.body = nn.Sequential(nn.Linear(f, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(512, h), nn.BatchNorm1d(h), nn.SiLU())
        self.gate = nn.Linear(h, n_b)
        self.exp = nn.ModuleList([nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1)) for _ in range(n_b)])
    def forward(self, x):
        z = self.body(x)
        return torch.cat([torch.softmax(self.gate(z), -1), torch.cat([e(z) for e in self.exp], 1)], 1)

# ------------------------------------------------------------------ losses
def twohead_loss(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()
twohead_pred = lambda o: torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)
mse_loss = lambda o, y: ((o[:, 0] - torch.log1p(y)) ** 2).mean()
mse_pred = lambda o: o[:, 0]
EDGES = torch.tensor([0.0, 1e-6, 2.5, 3.5, 4.3, 5.0, 5.8, 6.9])
def mdme_loss(o, y):
    g, v = o[:, :8], o[:, 8:]
    ly = torch.log1p(y)
    b = torch.clamp(torch.bucketize(ly, EDGES.to(y.device)) - 1, 0, 7)
    return (nn.functional.cross_entropy(torch.log(g + 1e-8), b)
            + ((v.gather(1, b.unsqueeze(1)).squeeze(1) - ly) ** 2).mean())
mdme_pred = lambda o: (o[:, :8] * torch.clamp(o[:, 8:], min=0)).sum(1)

# ------------------------------------------------------------------ training
def train_seq(mk_enc, weekly, tr, ev, epochs, seeds, bs=2048, lr=1.5e-3):
    mu, sd = norm_stats(tr)
    TABS = {a: tab(a, mu, sd) for a in tr}; YT = {a: Y[a].astype(np.float32) for a in tr}
    TE = tab(ev, mu, sd); outs = []
    for s in range(seeds):
        torch.manual_seed(s)
        model = SeqHybrid(mk_enc(C), FDIM).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        steps = epochs * len(tr) * (NU // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10))
        rng = np.random.default_rng(s)
        for _ in range(epochs):
            model.train()
            for a in tr:
                perm = rng.permutation(NU)
                for i in range(0, NU, bs):
                    idx = np.sort(perm[i:i + bs])
                    if len(idx) < 64: continue
                    opt.zero_grad(set_to_none=True)
                    loss = twohead_loss(model(torch.from_numpy(seq_slice(a, idx, weekly)).to(DEV),
                                              torch.from_numpy(TABS[a][idx]).to(DEV)),
                                        torch.from_numpy(YT[a][idx]).to(DEV))
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                    if sch.last_epoch < steps - 1: sch.step()
        model.eval(); acc = []
        with torch.no_grad():
            for i in range(0, NU, 8192):
                idx = np.arange(i, min(i + 8192, NU))
                o = model(torch.from_numpy(seq_slice(ev, idx, weekly)).to(DEV),
                          torch.from_numpy(TE[idx]).to(DEV))
                acc.append(twohead_pred(o).float().cpu())
        outs.append(torch.cat(acc).numpy()); del model; torch.cuda.empty_cache()
    return np.mean(outs, 0)

def train_tab(mk_model, loss_fn, pred_fn, tr, ev, epochs, seeds, bs=8192, lr=1e-3):
    mu, sd = norm_stats(tr)
    Ztr = np.concatenate([tab(a, mu, sd) for a in tr], 0)
    ytr = np.concatenate([Y[a] for a in tr]).astype(np.float32)
    Zev = tab(ev, mu, sd); outs = []
    Xt = torch.from_numpy(Ztr); yt = torch.from_numpy(ytr); n = len(Ztr)
    for s in range(seeds):
        torch.manual_seed(s)
        model = mk_model(FDIM).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs + 1))
        for _ in range(epochs):
            model.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 32: continue
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xt[idx].to(DEV)), yt[idx].to(DEV))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sch.step()
        model.eval(); acc = []
        with torch.no_grad():
            Ze = torch.from_numpy(Zev)
            for i in range(0, NU, 32768):
                acc.append(pred_fn(model(Ze[i:i + 32768].to(DEV))).float().cpu())
        outs.append(torch.cat(acc).numpy()); del model; torch.cuda.empty_cache()
    return np.mean(outs, 0)

def emit(logp, name):
    lp = np.clip(np.nan_to_num(logp), 0, 13)
    lp = lp - lp.mean() + EL
    p = np.clip(np.expm1(lp), 0, None)
    uids = np.load(CACHE / "uids.npy")
    with open(SUB / f"{name}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, p):
            f.write(f"{u},{v:.6f}\n")
    print(f"  -> {name}.csv  mean={p.mean():.2f} p50={np.percentile(p,50):.2f} "
          f"p99={np.percentile(p,99):.0f} max={p.max():.0f}")

JOBS = [
    ("gru_hybrid", "seq", lambda c: GRUEnc(c), True, 6),
    ("patchtst_hybrid", "seq", lambda c: PatchEnc(c, plen=14), False, 5),
    ("mlp_wide", "tab", lambda f: MLP(f, h=(1024, 512, 256), out=1), None, 16),
    ("mdme", "tab", lambda f: MDME(f), None, 14),
]
for name, kind, mk, weekly, ep in JOBS:
    print(f"\n=== {name} ===")
    try:
        t0 = time.time()
        if kind == "seq":
            pr = train_seq(mk, weekly, TR_SEQ_P, FIT, ep, 1)
            print(f"  probe RMSLE {rmsle(Y[FIT], np.expm1(np.clip(pr,0,13))):.5f}  ({time.time()-t0:.0f}s, 1 seed)")
            fin = train_seq(mk, weekly, TR_SEQ_F, PRED, ep, SEEDS)
        else:
            lf, pf = (mse_loss, mse_pred) if name == "mlp_wide" else (mdme_loss, mdme_pred)
            pr = train_tab(mk, lf, pf, TR_TAB_P, FIT, ep, 1)
            print(f"  probe RMSLE {rmsle(Y[FIT], np.expm1(np.clip(pr,0,13))):.5f}  ({time.time()-t0:.0f}s, 1 seed)")
            fin = train_tab(mk, lf, pf, TR_TAB_F, PRED, ep, SEEDS)
        print(f"  final done ({time.time()-t0:.0f}s total)")
        emit(fin, name)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
print("\ndone")
