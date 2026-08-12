"""Shared harness for the architecture zoo.

One feature matrix, one training protocol, one submission writer, so that every architecture is
compared on identical footing. Two fits per model: a probe at 2026-01-14 (sanity check that the
thing trained at all) and the real fit whose prediction goes to 2026-02-13.

The go/no-go decision is not made on the probe — the leaderboard has shown four times that a single
validation window does not order models. It is made on the stack threshold
`corr(L, new) > corr(L, stack) * g`, where g is computable without any submission.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "models")
from common import N_DAYS, NU, target, rmsle, build_features, OUT, CACHE, ds
from features_v2 import build_v2

PRED, FIT = N_DAYS - 1, 378
TR_PROBE = [348, 320, 292, 264]
TR_FINAL = [378, 350, 322, 294]
ALL_A = sorted(set(TR_PROBE + TR_FINAL + [PRED, FIT]))
EL = 2.3284
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)

_S = {}

def load():
    """returns (Xz dict of standardised float32 matrices, y dict, names, top-K index by gain)"""
    if _S:
        return _S["X"], _S["Y"], _S["names"], _S["top"]
    X1, N1 = build_features(ALL_A)
    BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
    BTN = ["btyd_ex", "btyd_alive", "btyd_gg", "eb_p", "eb_logblk"]
    for a in ALL_A:
        X1[a] = np.concatenate([X1[a], BT[a]], 1)
    X2, N2 = build_v2(ALL_A)
    names = N1 + BTN + N2
    Xr = {a: np.concatenate([X1[a], X2[a]], 1) for a in ALL_A}
    del X1, X2

    # log1p + standardise on the probe-training anchors; the sd floor and the clip are required —
    # a couple of near-constant columns otherwise produce inputs of order 1e6 and diverge the nets
    Z = np.nan_to_num(np.concatenate([np.log1p(np.clip(Xr[a], 0, None)) for a in TR_PROBE], 0),
                      posinf=0, neginf=0)
    mu, sd = Z.mean(0), np.maximum(Z.std(0), 1e-2)
    del Z
    X = {a: np.clip((np.nan_to_num(np.log1p(np.clip(Xr[a], 0, None)), posinf=0, neginf=0) - mu) / sd,
                    -8, 8).astype(np.float32) for a in ALL_A}
    Y = {a: target(a) for a in ALL_A if a != PRED}

    # feature ordering by gain — the attention models take only the top slice, otherwise the
    # token sequence is 296 long and attention becomes the whole compute budget
    import lightgbm as lgb
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63, max_bin=63,
                          n_jobs=23, verbose=-1, random_state=0, force_col_wise=True)
    g.fit(np.concatenate([X[a] for a in TR_PROBE], 0),
          np.concatenate([np.log1p(Y[a]) for a in TR_PROBE]))
    top = np.argsort(g.booster_.feature_importance("gain"))[::-1]
    _S.update(X=X, Y=Y, names=names, top=top, raw=Xr)
    print(f"zoo: {X[PRED].shape[1]} features, anchors {ALL_A}")
    return X, Y, names, top

def emit(logp, name):
    """level-match to the leaderboard-solved E[log1p(y)] and write a submission"""
    lp = np.clip(np.nan_to_num(np.asarray(logp, dtype=np.float64)), 0, 13)
    lp = lp - lp.mean() + EL
    q = np.clip(np.expm1(lp), 0, None)
    uids = np.load(CACHE / "uids.npy")
    with open(SUB / f"{name}.csv", "w") as f:
        f.write("user_id,predict\n")
        for u, v in zip(uids, q):
            f.write(f"{u},{v:.6f}\n")
    print(f"    -> {name}.csv  mean={q.mean():.2f} p50={np.percentile(q,50):.2f} "
          f"p99={np.percentile(q,99):.0f} max={q.max():.0f}")
    return q

# ------------------------------------------------------------------ torch trainer
def torch_fit_predict(make_model, loss_fn, pred_fn, Xtr, ytr, Xev, *, epochs=12, bs=8192,
                      lr=1e-3, wd=1e-5, dev="cuda", clip=5.0, seeds=1, extra=None):
    """generic loop shared by every neural model in the zoo; `pred_fn` returns log1p-scale output"""
    import torch
    outs = []
    n = len(Xtr)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.float32))
    for s in range(seeds):
        torch.manual_seed(s)
        model = make_model(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        steps = epochs * (n // bs + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10))
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 64:
                    continue
                xb = Xt[idx].to(dev, non_blocking=True)
                yb = yt[idx].to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb) if extra is None else loss_fn(model(xb), yb, extra)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
        model.eval()
        acc = []
        with torch.no_grad():
            Xe = torch.from_numpy(Xev)
            for i in range(0, len(Xev), 16384):
                acc.append(pred_fn(model(Xe[i:i + 16384].to(dev))).float().cpu())
        outs.append(np.concatenate([a.numpy() for a in acc]))
        del model
        torch.cuda.empty_cache()
    return np.mean(outs, 0)

def run_pair(label, fname, fit_fn, cols=None, verbose=True):
    """fit_fn(Xtr, ytr, Xev) -> log1p-scale prediction. Probe at FIT, then the real fit for PRED."""
    X, Y, _, _ = load()
    c = slice(None) if cols is None else cols
    t0 = time.time()
    try:
        Xtr = np.concatenate([X[a][:, c] for a in TR_PROBE], 0)
        ytr = np.concatenate([Y[a] for a in TR_PROBE])
        lp = fit_fn(Xtr, ytr, X[FIT][:, c])
        pr = rmsle(Y[FIT], np.expm1(np.clip(np.nan_to_num(lp), 0, 13)))
        del Xtr
        if verbose:
            print(f"    probe {ds(FIT)}: {pr:.5f}  ({time.time()-t0:.0f}s)")
        Xtr = np.concatenate([X[a][:, c] for a in TR_FINAL], 0)
        ytr = np.concatenate([Y[a] for a in TR_FINAL])
        lf = fit_fn(Xtr, ytr, X[PRED][:, c])
        del Xtr
        emit(lf, fname)
        return pr
    except Exception as e:
        import traceback
        print(f"    FAILED {label}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None
