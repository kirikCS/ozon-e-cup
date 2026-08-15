"""CoLES: self-supervised contrastive representation learning on the daily behaviour sequences.

Borrowed from the data_fusion fraud pipeline, where it was the single largest reported gain. The
assumption: two random sub-sequences of the same client should embed closer to each other than to
sub-sequences of any other client. Training uses no labels at all, so the embeddings cannot leak —
unlike hidden states pulled from a supervised model, which would need leave-one-anchor-out.

That property is what makes this worth doing here: it is the only way to add a learned
representation as a plain feature without a leakage story.

Pipeline: encoder over the 7-channel daily panel -> InfoNCE over user-matched window pairs ->
embeddings at each anchor from the window ending there -> appended to the 296 tabular features ->
LightGBM and CatBoost.
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
W = 182                 # window length fed to the encoder
DIM = 96                # embedding size
EPOCHS = 6
BS = 512                # users per contrastive batch
TEMP = 0.1

X, Y, NAMES, TOP = load()

print("building sequence tensor ...")
SEQ = np.stack([np.log1p(panel("gmv")).astype(np.float16),
                np.log1p(panel("gmv_search").astype(np.float32)).astype(np.float16),
                np.log1p(panel("ord").astype(np.float32)).astype(np.float16),
                np.log1p(panel("cart").astype(np.float32)).astype(np.float16),
                np.log1p(panel("srch").astype(np.float32)).astype(np.float16),
                panel("visit").astype(np.float16),
                panel("engaged").astype(np.float16)], 1)
C = SEQ.shape[1]
print(f"  SEQ {SEQ.shape}")

def window(users, end):
    """channel window [end-W+1, end] for the given users, zero-padded on the left"""
    s = end - W + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), C, -s), np.float32), got], 2)

class Encoder(nn.Module):
    """dilated causal CNN trunk plus a GRU read-out, the shape pytorch-lifestream uses for
    event sequences, sized down for a 182-step daily panel"""
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
        r, _ = self.rnn(z.transpose(1, 2))
        e = torch.cat([z.mean(-1), r[:, -1]], -1)
        return Fn.normalize(self.proj(e), dim=-1)

def info_nce(a, b, temp=TEMP):
    """two views of the same client are positives; every other client in the batch is a negative"""
    logits = a @ b.T / temp
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (Fn.cross_entropy(logits, tgt) + Fn.cross_entropy(logits.T, tgt))

def train_encoder(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(C).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3, weight_decay=1e-5)
    steps_per_epoch = NU // BS
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                              total_steps=EPOCHS * steps_per_epoch)
    for ep in range(EPOCHS):
        enc.train()
        perm = rng.permutation(NU)
        run = 0.0
        for i in range(0, NU - BS, BS):
            users = np.sort(perm[i:i + BS])
            # two independent window ends per client, at least 30 days apart where possible
            e1 = rng.integers(W // 2, N_DAYS, size=1)[0]
            e2 = rng.integers(W // 2, N_DAYS, size=1)[0]
            x1 = torch.from_numpy(window(users, int(e1))).to(DEV)
            x2 = torch.from_numpy(window(users, int(e2))).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = info_nce(enc(x1), enc(x2))
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step(); sch.step()
            run += loss.item()
        print(f"    epoch {ep}: InfoNCE {run / max(1, steps_per_epoch):.4f}")
    return enc

@torch.no_grad()
def embed(enc, anchor):
    enc.eval()
    out = np.empty((NU, DIM), np.float32)
    for i in range(0, NU, 8192):
        u = np.arange(i, min(i + 8192, NU))
        out[u] = enc(torch.from_numpy(window(u, anchor)).to(DEV)).float().cpu().numpy()
    return out

t0 = time.time()
print("training CoLES encoder (без меток) ...")
ENC = train_encoder(0)
print(f"  {time.time()-t0:.0f}s")

ANCHORS = sorted(set(TR_PROBE + TR_FINAL + [FIT, PRED]))
EMB = {a: embed(ENC, a) for a in ANCHORS}
np.save(CACHE / "coles_emb.npy", np.stack([EMB[a] for a in ANCHORS]))
print(f"эмбеддинги готовы: {DIM} измерений на якорь")

# quick sanity: do the embeddings carry target signal at all?
from sklearn.linear_model import Ridge
r = Ridge(alpha=10.0).fit(EMB[TR_PROBE[0]], np.log1p(Y[TR_PROBE[0]]))
pr = r.predict(EMB[FIT])
print(f"линейный зонд по одним эмбеддингам: RMSLE {rmsle(Y[FIT], np.expm1(np.clip(pr,0,13))):.5f}")

import lightgbm as lgb
from catboost import CatBoostRegressor

LGB = dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True)

def with_emb(a):
    return np.concatenate([X[a], EMB[a]], 1)

def run(name, fname, fitter):
    Xtr = np.concatenate([with_emb(a) for a in TR_PROBE], 0)
    ytr = np.concatenate([Y[a] for a in TR_PROBE])
    p = fitter(Xtr, np.log1p(ytr), with_emb(FIT)); del Xtr
    print(f"  {name}: probe {rmsle(Y[FIT], np.clip(p,0,None)):.5f}")
    Xtr = np.concatenate([with_emb(a) for a in TR_FINAL], 0)
    ytr = np.concatenate([Y[a] for a in TR_FINAL])
    pf = fitter(Xtr, np.log1p(ytr), with_emb(PRED)); del Xtr
    emit(np.log1p(np.clip(pf, 0, None)), fname)

def f_lgb(Xtr, lytr, Xev):
    m = lgb.LGBMRegressor(**dict(LGB, objective="regression")); m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev))

def f_cat(Xtr, lytr, Xev):
    m = CatBoostRegressor(iterations=800, learning_rate=0.09, depth=7, l2_leaf_reg=5.0,
                          border_count=64, loss_function="RMSE", verbose=0, thread_count=23,
                          random_seed=42, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(Xtr, lytr)
    return np.expm1(m.predict(Xev))

print("\nбустинг с эмбеддингами CoLES:")
run("LightGBM + CoLES", "coles_lgbm", f_lgb)
run("CatBoost + CoLES", "coles_catboost", f_cat)
print("\ndone")
