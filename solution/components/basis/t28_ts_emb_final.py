"""ts_emb FINAL: two_stage_v3 + CoLES-эмбеддинги (v1, 64-d), 9 сидов, уровень EL.

A/B дал -0.00045 при corr(A,B) 0.99826 - ниже порога компонента, но это лучший
оставшийся самодельный кандидат: в стеке оценка ~1e-4, и его ценность вырастет при
перераспределении весов после вшивки вечерней нейронки. Сборка бесплатная; тратить ли
твинк-замер - решение отдельное.

Output: submissions/ts_emb_9seed.csv
"""
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, N_DAYS, panel, target, rmsle, build_features, OUT, CACHE

DEV = "cuda"
PRED = N_DAYS - 1
TR_FINAL = [378, 350, 322, 294, 266, 238, 210, 182, 154, 126]
ANCH = TR_FINAL + [PRED]
SEEDS = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
DROP = {"tgt_month", "tgt_doy_sin", "tgt_doy_cos", "hist_days"}
K, EMB = 128, 64
EL = 2.3284
WD0 = date(2025, 1, 1).weekday()
assert torch.cuda.is_available()

LGB = dict(n_estimators=1200, learning_rate=0.04, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, force_col_wise=True)

G = panel("gmv"); S = panel("srch"); CA = panel("cart"); OR = panel("ord")
V = panel("visit"); SC = panel("scart"); SO = panel("sord")
SD = panel("sday"); CD = panel("catday")


def build_events(a):
    mask = V[:, :a + 1] > 0
    rev = mask[:, ::-1]
    cc = rev.cumsum(1)
    take = rev & (cc <= K)
    ui, jj = np.nonzero(take)
    day = a - jj
    slot = K - cc[ui, jj]
    E = np.zeros((NU, K, 12), np.float16)
    E[ui, slot, 0] = np.log1p(G[ui, day])
    E[ui, slot, 1] = np.log1p(S[ui, day].astype(np.float32))
    E[ui, slot, 2] = np.minimum(CA[ui, day], 20)
    E[ui, slot, 3] = np.minimum(OR[ui, day], 10)
    E[ui, slot, 5] = np.log1p((a - day).astype(np.float32))
    wd = (day + WD0) % 7
    E[ui, slot, 6] = np.sin(2 * np.pi * wd / 7)
    E[ui, slot, 7] = np.cos(2 * np.pi * wd / 7)
    E[ui, slot, 8] = np.minimum(SC[ui, day], 20)
    E[ui, slot, 9] = np.minimum(SO[ui, day], 10)
    E[ui, slot, 10] = SD[ui, day]
    E[ui, slot, 11] = CD[ui, day]
    prev = np.full((NU, K), -1, np.int64)
    prev[ui, slot] = day
    d0 = prev.astype(np.float64)
    valid = prev >= 0
    dshift = np.roll(d0, 1, axis=1); vshift = np.roll(valid, 1, axis=1)
    dshift[:, 0] = -1; vshift[:, 0] = False
    g = np.where(valid & vshift, d0 - dshift, 0)
    E[:, :, 4] = np.log1p(np.clip(g, 0, None)).astype(np.float16)
    return E


class Enc(nn.Module):
    def __init__(self, f=12, h=256, e=EMB):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, e))
    def forward(self, x):
        z = torch.nn.functional.silu(self.inp(x))
        o, _ = self.gru(z)
        return torch.nn.functional.normalize(self.proj(o[:, -1]), dim=1)


print("extract embeddings (v1) for FINAL anchors...", flush=True)
enc = Enc().to(DEV)
enc.load_state_dict(torch.load(OUT / "coles_enc.pt"))
enc.eval()
EMBS = {}
for a in ANCH:
    p = OUT / f"coles_emb_a{a}.npy"
    if p.exists():
        EMBS[a] = np.load(p)
        continue
    t0 = time.time()
    Ea = build_events(a)
    acc = []
    with torch.no_grad():
        for i in range(0, NU, 8192):
            x = torch.from_numpy(Ea[i:i + 8192].astype(np.float32)).to(DEV)
            acc.append(enc(x).float().cpu().numpy())
    EMBS[a] = np.concatenate(acc)
    np.save(p, EMBS[a])
    print(f"  anchor {a} ({time.time()-t0:.0f}s)", flush=True)
    del Ea

BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
assert all(a in BT for a in ANCH), "BTYD missing anchors"
X, NAMES = build_features(ANCH, verbose=False)
keep = [i for i, n in enumerate(NAMES) if n not in DROP]
for a in ANCH:
    X[a] = np.concatenate([X[a][:, keep], BT[a], EMBS[a].astype(np.float32)], 1)
Y = {a: target(a) for a in ANCH if a != PRED}
print(f"{X[PRED].shape[1]} features", flush=True)

finals = []
for seed in SEEDS:
    t0 = time.time()
    Xtr = np.concatenate([X[a] for a in TR_FINAL], 0)
    lytr = np.concatenate([np.log1p(Y[a]) for a in TR_FINAL])
    pos = lytr > 0
    clf = lgb.LGBMClassifier(**LGB, random_state=seed)
    clf.fit(Xtr, pos.astype(np.int8))
    pb = clf.predict_proba(X[PRED])[:, 1]
    reg = lgb.LGBMRegressor(**LGB, objective="regression", random_state=seed)
    reg.fit(Xtr[pos], lytr[pos])
    finals.append(pb * np.clip(reg.predict(X[PRED]), 0, None))
    del Xtr, lytr
    print(f"  final seed {seed:>5}: done ({time.time()-t0:.0f}s)", flush=True)

lf = np.mean(finals, 0)
lf = lf - lf.mean() + EL
pred = np.clip(np.expm1(lf), 0, None)
uids = np.load(CACHE / "uids.npy")
with open("submissions/ts_emb_9seed.csv", "w") as f:
    f.write("user_id,predict\n")
    for u, v in zip(uids, pred):
        f.write(f"{u},{v:.6f}\n")
print(f"ts_emb_9seed.csv: mean log1p={np.log1p(pred).mean():.4f}")

import os
os._exit(0)
