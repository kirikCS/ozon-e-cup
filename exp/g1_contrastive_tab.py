"""Contrastive encoder over the tabular columns, its embedding fed to the daily generative model.

The addressee matters more than the encoder. CoLES already did contrastive learning here — InfoNCE
over two windows of the same client, on the RAW daily panels. Its linear probe scored 1.73703 on
its own, which is strong, and adding it to boosting gave nothing: the embeddings duplicated what
the window aggregates already carry. That is the trap to avoid repeating.

So this does not feed seq_hybrid. That model already has all 165 tabular columns, and handing it a
64-dim compression of them would be strictly less information. It feeds the DAILY GENERATIVE
MODEL, which by construction sees no tabular feature at all — only the raw series and the calendar
of the horizon. A compact summary is new information there, and being compact is the point:
passing all 165 columns is what turned seq_pure (corr 0.9856 with the tree cluster) into
seq_hybrid (0.9944).

Positives are the SAME CLIENT AT TWO DIFFERENT ANCHORS, at least 56 days apart. That asks the
encoder for what persists about a client across time, which is exactly the part worth carrying
into a model that reads a different window. Feature-corruption positives (SCARF style) are the
control: they ask only for invariance to noise, with no notion of client identity.

  infonce   same client, two anchors, InfoNCE over the batch
  triplet   same pairs, margin triplet loss with semi-hard negatives
  scarf     same client, one anchor, positives by corrupting 30% of columns — the control

Memory: this box has 36 GB against the old one's 98, so the feature store is built once for the
anchors actually used and the daily panels are never held alongside it.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import build_features, target, rmsle, NU, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM, EPOCHS, BS, LR, TEMP = 64, 12, 1024, 2e-3, 0.1
ANCHORS = [318, 290, 262, 234, 206, 348]
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

print("строю фичестор ...", flush=True)
t0 = time.time()
X, names = build_features(ANCHORS, verbose=False)
F = len(names)
MU = np.mean([X[a].mean(0) for a in ANCHORS], 0)
SD = np.maximum(np.mean([X[a].std(0) for a in ANCHORS], 0), 1e-2)
Z = {a: np.clip((X[a] - MU) / SD, -8, 8).astype(np.float32) for a in ANCHORS}
del X
print(f"  {F} признаков, {time.time()-t0:.0f}s", flush=True)

class Enc(nn.Module):
    def __init__(self, f, d=DIM, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.BatchNorm1d(h), nn.SiLU(),
                                 nn.Linear(h, d))
    def forward(self, x):
        return Fn.normalize(self.net(x), dim=-1)

def info_nce(a, b, temp=TEMP):
    logits = a @ b.T / temp
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (Fn.cross_entropy(logits, tgt) + Fn.cross_entropy(logits.T, tgt))

def triplet(a, b, margin=0.3):
    """semi-hard: hardest negative that is still farther than the positive"""
    pos = (a * b).sum(-1)
    sim = a @ b.T
    eye = torch.eye(len(a), device=a.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -2.0)
    mask = sim < pos.unsqueeze(1)
    sim_semi = torch.where(mask, sim, torch.full_like(sim, -2.0))
    neg = torch.where(mask.any(1), sim_semi.max(1).values, sim.max(1).values)
    return Fn.relu(margin - pos + neg).mean()

def train(mode, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Enc(F).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    tr = SEL_TR
    for ep in range(EPOCHS):
        m.train()
        perm = rng.permutation(NU)
        run = nb = 0
        for i in range(0, NU - BS, BS):
            idx = np.sort(perm[i:i + BS])
            if mode == "scarf":
                a = tr[rng.integers(len(tr))]
                x1 = Z[a][idx].copy()
                x2 = x1.copy()
                cm = rng.random(x2.shape) < 0.30
                shuffle = x2[rng.permutation(len(idx))]
                x2[cm] = shuffle[cm]
            else:
                # два якоря одного клиента, не ближе 56 дней друг к другу
                while True:
                    a1, a2 = tr[rng.integers(len(tr))], tr[rng.integers(len(tr))]
                    if abs(a1 - a2) >= 56:
                        break
                x1, x2 = Z[a1][idx], Z[a2][idx]
            e1 = m(torch.from_numpy(np.ascontiguousarray(x1)).to(DEV))
            e2 = m(torch.from_numpy(np.ascontiguousarray(x2)).to(DEV))
            loss = info_nce(e1, e2) if mode in ("infonce", "scarf") else triplet(e1, e2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            sch.step()
            run += loss.item(); nb += 1
        if ep in (0, EPOCHS // 2, EPOCHS - 1):
            print(f"    {mode} эпоха {ep}: loss {run/max(nb,1):.4f}", flush=True)
    m.eval()
    emb = {}
    with torch.no_grad():
        for a in ANCHORS:
            out = []
            for i in range(0, NU, 16384):
                out.append(m(torch.from_numpy(Z[a][i:i + 16384]).to(DEV)).cpu().numpy())
            emb[a] = np.concatenate(out)
    del m
    torch.cuda.empty_cache()
    return emb

from sklearn.linear_model import Ridge
y = target(SEL_EV)
ly = np.log1p(y)
print(f"\n{'режим':>9} {'зонд Ridge':>11} {'InfoNCE-точность':>17} {'сек':>7}")
EMB = {}
for mode in ("infonce", "triplet", "scarf"):
    t0 = time.time()
    e = train(mode)
    EMB[mode] = e
    r = Ridge(alpha=10.0).fit(e[SEL_TR[0]], np.log1p(target(SEL_TR[0])))
    pr = r.predict(e[SEL_EV])
    # насколько эмбеддинг узнаёт клиента: доля, где ближайший на другом якоре — он сам
    sub = np.arange(0, NU, 50)
    A = e[SEL_TR[0]][sub]; B = e[SEL_TR[-1]][sub]
    acc = float((np.argmax(A @ B.T, 1) == np.arange(len(sub))).mean())
    print(f"{mode:>9} {rmsle(y, np.expm1(np.clip(pr,0,13))):11.5f} {acc:17.4f} "
          f"{time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"g1_emb_{mode}.npy", np.stack([e[a] for a in ANCHORS]))
print("\nвторая колонка — доля клиентов, у которых ближайший сосед на другом якоре это он сам")
print("(случайное угадывание дало бы 0.0002); высокая доля значит, что энкодер выучил личность,")
print("а не просто сжал признаки")
print("\nэмбеддинги сохранены; следующий шаг — подать их в дневную генеративную модель")
print("\ndone")
