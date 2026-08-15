"""Hidden representations and logits of the encoder as features for boosting, without leakage.

The naive version of this is wrong. If the encoder is trained on anchors A and its hidden states at
those same anchors are then handed to a GBDT that also trains on A, the states are in-sample: they
already contain the labels the GBDT is trying to predict, so the GBDT learns to trust them far more
than it should, and the trust does not survive to a fresh anchor.

Leave-one-anchor-out fixes it. For every training anchor a, the encoder that produces the features
at a was fitted without a. The evaluation anchor is never in any training set, so its features come
from the encoder fitted on everything. This is the walk-forward OOF construction from the
data_fusion pipeline, with the anchor playing the role of the time fold.

Three variations, to separate what actually carries the signal:
    logits  the two head outputs only (P(y>0) logit and the conditional log1p mean) — 2 columns
    hidden  the penultimate representation — 128 columns, no explicit prediction
    both    logits + hidden
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "models")
from zoo_common import load, emit, TR_PROBE, TR_FINAL, FIT, PRED
from common import NU, rmsle, ds, CACHE
import t22_encoders as E          # reuse Model / win / loss_fn / pred_fn

DEV = E.DEV
X, Y, NAMES, TOP = load()
KIND, FUSION = "gated", "gated-bi"          # the best of the five encoder variants
EPOCHS = 5

def fit_encoder(anchors, seed=0):
    torch.manual_seed(seed)
    m = E.Model(E.FDIM, KIND, FUSION).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-5)
    bs = 2048
    steps = EPOCHS * len(anchors) * (NU // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=max(steps, 10))
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        m.train()
        for a in anchors:
            perm = rng.permutation(NU)
            for i in range(0, NU, bs):
                idx = np.sort(perm[i:i + bs])
                if len(idx) < 64:
                    continue
                xs = torch.from_numpy(E.win(idx, a)).to(DEV)
                xt = torch.from_numpy(X[a][idx]).to(DEV)
                yb = torch.from_numpy(Y[a][idx].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                E.loss_fn(m(xs, xt), yb).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    return m.eval()

@torch.no_grad()
def represent(m, anchor):
    """penultimate activations (128) and the two head logits (2) for every user at this anchor"""
    hid, log = [], []
    grab = {}
    h = m.head[2].register_forward_hook(lambda mod, i, o: grab.__setitem__("h", o.detach()))
    for i in range(0, NU, 8192):
        idx = np.arange(i, min(i + 8192, NU))
        o = m(torch.from_numpy(E.win(idx, anchor)).to(DEV), torch.from_numpy(X[anchor][idx]).to(DEV))
        hid.append(grab["h"].float().cpu().numpy())
        log.append(o.float().cpu().numpy())
    h.remove()
    return np.concatenate(hid), np.concatenate(log)

def oof_features(train_anchors, eval_anchor):
    """LOO representations at every training anchor; full-fit representations at the eval anchor"""
    H, L = {}, {}
    for a in train_anchors:
        rest = [b for b in train_anchors if b != a]
        t0 = time.time()
        m = fit_encoder(rest, seed=hash(a) % 1000)
        H[a], L[a] = represent(m, a)
        del m
        torch.cuda.empty_cache()
        print(f"    LOO без якоря {a}: {time.time()-t0:.0f}s")
    m = fit_encoder(train_anchors, seed=7)
    H[eval_anchor], L[eval_anchor] = represent(m, eval_anchor)
    del m
    torch.cuda.empty_cache()
    return H, L

import lightgbm as lgb
LGB = dict(n_estimators=700, learning_rate=0.07, num_leaves=63, min_child_samples=100,
           max_bin=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           n_jobs=23, verbose=-1, random_state=42, force_col_wise=True, objective="regression")

def block(a, H, L, mode):
    parts = [X[a]]
    if mode in ("logits", "both"):
        parts.append(L[a])
    if mode in ("hidden", "both"):
        parts.append(H[a])
    return np.concatenate(parts, 1).astype(np.float32)

def stage(train_anchors, eval_anchor, tag):
    print(f"  {tag}: строю OOF-представления ...")
    H, L = oof_features(train_anchors, eval_anchor)
    out = {}
    for mode in ("logits", "hidden", "both"):
        Xtr = np.concatenate([block(a, H, L, mode) for a in train_anchors], 0)
        ytr = np.log1p(np.concatenate([Y[a] for a in train_anchors]))
        m = lgb.LGBMRegressor(**LGB).fit(Xtr, ytr)
        del Xtr
        out[mode] = m.predict(block(eval_anchor, H, L, mode))
        n_extra = block(eval_anchor, H, L, mode).shape[1] - X[eval_anchor].shape[1]
        print(f"    {mode:7s} (+{n_extra} колонок): "
              + (f"probe {rmsle(Y[eval_anchor], np.expm1(np.clip(out[mode],0,13))):.5f}"
                 if eval_anchor == FIT else "final"))
    return out

print("=== OOF: скрытые представления и логиты энкодера как признаки ===")
p = stage(TR_PROBE, FIT, "probe")
f = stage(TR_FINAL, PRED, "final")
for mode in ("logits", "hidden", "both"):
    emit(np.clip(f[mode], 0, 13), f"oof_{mode}")
print("\ndone")
