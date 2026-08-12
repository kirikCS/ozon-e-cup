"""Tier 5: sequence models over the raw daily behaviour panel (GPU).
Tokenised daily/weekly user behaviour -> encoder -> the exact two-head RMSLE factorisation.
Encoders: dilated causal CNN (WaveNet-style), GRU, PatchTST-style transformer, plus hybrids
that concatenate the tabular feature vector at the head.
"""
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from common import (VAL_ANCHOR, HON_ANCHOR, NU, panel, target, cohort_rule, rmsle, record,
                    Timer, build_features, OUT)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)
A, AH = VAL_ANCHOR, HON_ANCHOR
TR_VAL = [348, 320, 292, 264]
TR_HON = [258, 230, 202, 174]
ALL_A = sorted(set(TR_VAL + TR_HON + [A, AH]))
L_DAY = 182                     # days of history fed to the encoder (26 weeks)
W = 7                           # week length for the aggregated views

# ------------------------------------------------------------------ tensors
print("building sequence tensor ...")
chans = [np.log1p(panel("gmv")).astype(np.float16),
         np.log1p(panel("gmv_search").astype(np.float32)).astype(np.float16),
         np.log1p(panel("ord").astype(np.float32)).astype(np.float16),
         np.log1p(panel("cart").astype(np.float32)).astype(np.float16),
         np.log1p(panel("srch").astype(np.float32)).astype(np.float16),
         panel("visit").astype(np.float16),
         panel("engaged").astype(np.float16)]
SEQ = np.stack(chans, axis=1)                   # (NU, C, T)
C = SEQ.shape[1]
print("  SEQ", SEQ.shape, SEQ.nbytes / 1e9, "GB")

X, NAMES = build_features(ALL_A)
try:
    BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
    for a in ALL_A:
        if a in BT: X[a] = np.concatenate([X[a], BT[a]], 1)
except Exception:
    pass
FDIM = X[ALL_A[0]].shape[1]
Y = {a: target(a) for a in ALL_A}
m_hon = cohort_rule(AH)

# standardise the tabular block. The sd floor and the output clip are required: without them a
# couple of near-constant columns produce inputs of order 1e6 and the hybrid heads diverge.
_Z = np.nan_to_num(np.concatenate([np.log1p(np.clip(X[a], 0, None)) for a in TR_VAL], 0),
                   posinf=0, neginf=0)
TMU, TSD = _Z.mean(0), np.maximum(_Z.std(0), 1e-2)
del _Z
def tab(a):
    z = np.nan_to_num(np.log1p(np.clip(X[a], 0, None)), posinf=0, neginf=0)
    return np.clip((z - TMU) / TSD, -8, 8).astype(np.float32)

def seq_slice(a, idx, weekly):
    """window [a-L_DAY+1, a], zero-padded on the left when the anchor has less history.
    Without the pad, early anchors produced a negative start index and silently empty slices."""
    s = a - L_DAY + 1
    if s >= 0:
        win = SEQ[idx, :, s:a + 1].astype(np.float32)
    else:
        got = SEQ[idx, :, 0:a + 1].astype(np.float32)
        win = np.concatenate([np.zeros((len(idx), C, -s), np.float32), got], axis=2)
    assert win.shape[2] == L_DAY, (win.shape, a)
    if weekly:
        win = win.reshape(win.shape[0], C, L_DAY // W, W).sum(-1)
    return win

PROTO = {"VAL": (TR_VAL, A, None), "HON": (TR_HON, AH, m_hon)}

# ------------------------------------------------------------------ encoders
class CNNEnc(nn.Module):
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
        return torch.cat([z.mean(-1), z.max(-1).values], -1)

class GRUEnc(nn.Module):
    def __init__(self, c, h=128):
        super().__init__()
        self.rnn = nn.GRU(c, h, num_layers=2, batch_first=True, dropout=0.1)
        self.out = h * 2
    def forward(self, x):
        z, _ = self.rnn(x.transpose(1, 2))
        return torch.cat([z[:, -1], z.mean(1)], -1)

class PatchEnc(nn.Module):
    """PatchTST-style: split the series into patches, embed, transformer, pool"""
    def __init__(self, c, plen=14, d=128, nl=3, nh=4):
        super().__init__()
        self.plen = plen
        self.emb = nn.Linear(c * plen, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, nh, d * 4, dropout=0.1, batch_first=True,
                                        norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(el, nl)
        self.out = d * 2
    def forward(self, x):
        B, c, T = x.shape
        n = T // self.plen
        p = x[:, :, :n * self.plen].reshape(B, c, n, self.plen).permute(0, 2, 1, 3).reshape(B, n, c * self.plen)
        z = self.emb(p) + self.pos[:, :n]
        z = self.tr(z)
        return torch.cat([z.mean(1), z[:, -1]], -1)

class Head(nn.Module):
    def __init__(self, d_seq, d_tab=0, h=256, drop=0.15):
        super().__init__()
        d = d_seq + d_tab
        self.body = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(h, 128), nn.BatchNorm1d(128), nn.SiLU())
        self.p = nn.Linear(128, 1); self.m = nn.Linear(128, 1)
    def forward(self, z):
        z = self.body(z)
        return torch.cat([self.p(z), self.m(z)], 1)

class SeqModel(nn.Module):
    def __init__(self, enc, d_tab=0):
        super().__init__()
        self.enc = enc
        self.head = Head(enc.out, d_tab)
        self.d_tab = d_tab
    def forward(self, xs, xt=None):
        z = self.enc(xs)
        if self.d_tab:
            z = torch.cat([z, xt], 1)
        return self.head(z)

def loss_fn(o, y):
    pos = (y > 0).float(); ly = torch.log1p(y)
    cls = nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos, reduction="none")
    return (cls + pos * (o[:, 1] - ly) ** 2).mean()

def pred_log(o):
    return torch.sigmoid(o[:, 0]) * torch.clamp(o[:, 1], min=0)

def train_eval(make_enc, weekly, use_tab, epochs=6, bs=2048, lr=1.5e-3, name=""):
    res, secs = {}, 0
    for key, (tr, ev, mask) in PROTO.items():
        with Timer() as t:
            d_tab = FDIM if use_tab else 0
            model = SeqModel(make_enc(C), d_tab).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            pairs = [(a, i) for a in tr for i in range(NU)]
            n = len(pairs)
            steps = epochs * (n // bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10))
            TABS = {a: tab(a) for a in tr} if use_tab else {}
            YT = {a: Y[a].astype(np.float32) for a in tr}
            rng = np.random.default_rng(0)
            for ep in range(epochs):
                model.train()
                for a in tr:
                    perm = rng.permutation(NU)
                    for i in range(0, NU, bs):
                        idx = np.sort(perm[i:i + bs])
                        if len(idx) < 64: continue
                        xs = torch.from_numpy(seq_slice(a, idx, weekly)).to(DEV)
                        yb = torch.from_numpy(YT[a][idx]).to(DEV)
                        xt = torch.from_numpy(TABS[a][idx]).to(DEV) if use_tab else None
                        opt.zero_grad(set_to_none=True)
                        loss = loss_fn(model(xs, xt), yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        if sched.last_epoch < steps - 1: sched.step()
            model.eval()
            TE = tab(ev) if use_tab else None
            outs = []
            with torch.no_grad():
                for i in range(0, NU, 8192):
                    idx = np.arange(i, min(i + 8192, NU))
                    xs = torch.from_numpy(seq_slice(ev, idx, weekly)).to(DEV)
                    xt = torch.from_numpy(TE[idx]).to(DEV) if use_tab else None
                    outs.append(pred_log(model(xs, xt)).float().cpu())
            p = np.expm1(np.clip(torch.cat(outs).numpy(), 0, None))
        secs += t.dt
        res[key] = rmsle(Y[ev], p) if mask is None else rmsle(Y[ev][mask], p[mask])
        if key == "VAL":
            np.save(OUT / f"pred_{name.replace(' ', '_')[:60]}.npy", p.astype(np.float32))
        del model
        torch.cuda.empty_cache()
    return res, secs

print("--- Tier 5: sequence models ---")
JOBS = [
    ("Dilated CNN over 364 daily steps", lambda c: CNNEnc(c), False, False, 5),
    ("GRU over 52 weekly steps", lambda c: GRUEnc(c), True, False, 6),
    ("PatchTST-style transformer (patch 14d)", lambda c: PatchEnc(c, plen=14), False, False, 5),
    ("Dilated CNN + tabular features (hybrid)", lambda c: CNNEnc(c), False, True, 5),
    ("PatchTST-style + tabular features (hybrid)", lambda c: PatchEnc(c, plen=14), False, True, 5),
    ("GRU weekly + tabular features (hybrid)", lambda c: GRUEnc(c), True, True, 6),
]
import os
ONLY = os.environ.get("ECUP_JOBS", "")
for name, mk, weekly, use_tab, ep in JOBS:
    if ONLY and ONLY.lower() not in name.lower():
        continue
    try:
        res, secs = train_eval(mk, weekly, use_tab, epochs=ep, name=name)
        record(name, "sequence", res["VAL"], res["HON"], secs,
               f"{'weekly' if weekly else 'daily'} tokens, two-head loss" + (", + tabular" if use_tab else ""))
    except Exception as e:
        print(f"  FAILED {name}: {type(e).__name__} {e}")
print("Tier 5 done")
