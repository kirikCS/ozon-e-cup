"""Daily generative model, round 3 (q2c): fix the dependence between days, then measure decorrelation.

Round 1 worked but two things were off.

LEVEL. Raw RMSLE 1.84588, but mean log1p of the prediction was 3.0022 against the target's
2.4170 — the model was massively over-levelled. Level-matched it is 1.74929, which is 0.007
behind the production LightGBM (1.74216) and clearly AHEAD of seq_pure (1.77127) at the same
anchor, while seeing none of the 165 tabular features. So the daily formulation is sound; the
level was the artefact.

DEPENDENCE. The diagnostic said why: simulated P(Y>0) = 0.6840 against an empirical 0.5552.
Sampling 30 days independently makes "at least one purchase in 30 days" far too likely, because
real purchases cluster. That single defect inflates the gate for everyone and is exactly what
pushed the level up.

The fix is the classic one for recurrent-event over-dispersion: a shared latent frailty per
client per path. One z ~ N(0,1) is drawn ONCE per simulated path and shifts every day's hazard
in that path, which induces positive dependence across days without touching the daily
marginals — those are preserved by the probit-style rescaling

    logit' = logit * sqrt(1 + pi*sigma^2/8)     since  E_z[sigmoid(a + sigma z)] ~ sigmoid(a / sqrt(1 + pi sigma^2/8))

sigma is a SINGLE parameter, fitted on the training anchors by matching simulated P(Y>0) to the
empirical one, then frozen — the same discipline every calibration in this project follows.

Also produces the FINAL vector at anchor 408 so the leaderboard decorrelation can be measured,
and reports the local correlation against a LightGBM trained at the same anchor.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, build_features, NU, N_DAYS, H, D0, ds, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 364
EPOCHS, BS, NPATH, GH = 8, 1024, 96, 16
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
FIN_TR, FIN_EV = [378, 350, 322, 294, 266], N_DAYS - 1
EL = 2.3284

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    P = panel(n).astype(np.float32)
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}")

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

def daily_y(users, a):
    return G[users, a + 1:a + 1 + H].astype(np.float32)

class Enc(nn.Module):
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

class Model(nn.Module):
    def __init__(self, c, d=128):
        super().__init__()
        self.enc = Enc(c)
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(self.enc.h * 3 + 10 + 16, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x, cal):
        h = self.enc(x)
        B = h.shape[0]
        z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                       cal.unsqueeze(0).expand(B, H, -1),
                       self.pos.unsqueeze(0).expand(B, H, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float()
    ly = torch.log1p(yd)
    bce = Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
    nll = 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)
    return (bce + pos * nll).sum(1).mean()

def fit(anchors, seed=0):
    torch.manual_seed(seed)
    m = Model(CH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(anchors) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in anchors}
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        m.train()
        run = nb = 0
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
def simulate(m, anchor, sigma, npath=NPATH, chunk=256, users=None):
    """frailty-coupled paths; the marginal daily hazard is preserved by the probit rescale"""
    cal = torch.from_numpy(cal_feats(anchor)).to(DEV)
    users = np.arange(NU) if users is None else users
    out = np.empty(len(users), np.float32)
    p0 = np.empty(len(users), np.float32)
    scale = float(np.sqrt(1 + np.pi * sigma ** 2 / 8))
    for i in range(0, len(users), chunk):
        idx = users[i:i + chunk]
        hz, mu, ls = m(torch.from_numpy(win(idx, anchor)).to(DEV), cal)
        hz = hz * scale
        sd = ls.exp()
        b = len(idx)
        acc = torch.zeros(b, device=DEV)
        acc0 = torch.zeros(b, device=DEV)
        # frailty интегрируется квадратурой Гаусса-Эрмита, а не разыгрывается: это главный
        # источник дисперсии оценки, и по нему интеграл берётся точно
        nodes, wts = np.polynomial.hermite_e.hermegauss(GH)
        wts = wts / wts.sum()
        for zn, wn in zip(nodes, wts):
            p = torch.sigmoid(hz + float(zn) * sigma)
            u = torch.rand(npath, b, H, device=DEV) < p.unsqueeze(0)
            v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                            * torch.randn(npath, b, H, device=DEV))
            tot = (u.float() * v.clamp(min=0)).sum(-1)
            acc += float(wn) * torch.log1p(tot).mean(0)
            acc0 += float(wn) * (tot > 0).float().mean(0)
        out[i:i + len(idx)] = acc.float().cpu().numpy()
        p0[i:i + len(idx)] = acc0.float().cpu().numpy()
    return out, p0

def fit_sigma(m, anchors):
    """single parameter, matched on the TRAINING anchors, then frozen"""
    sub = np.arange(0, NU, 12)
    emp = np.mean([np.mean(target(a)[sub] > 0) for a in anchors])
    print(f"  эмпирическая P(Y>0) на обучающих якорях: {emp:.4f}")
    best = None
    for s in (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5):
        got = np.mean([simulate(m, a, s, npath=32, users=sub)[1].mean() for a in anchors])
        print(f"    sigma {s:.1f} -> P(Y>0) {got:.4f}")
        if best is None or abs(got - emp) < best[0]:
            best = (abs(got - emp), s)
    print(f"  выбрана sigma = {best[1]}")
    return best[1]

def report(m, sigma, ev, tag):
    mc, p0 = simulate(m, ev, sigma)
    y = target(ev)
    ly = np.log1p(y)
    sh = mc - mc.mean() + ly.mean()
    print(f"  [{tag}] сырой {rmsle(y, np.expm1(np.clip(mc,0,13))):.5f}   "
          f"уровень выровнен {rmsle(y, np.expm1(np.clip(sh,0,13))):.5f}")
    print(f"  [{tag}] P(Y>0) модель {p0.mean():.4f} против эмпирической {(y>0).mean():.4f}")
    print(f"  [{tag}] corr(модель, log1p y) = {np.corrcoef(mc, ly)[0,1]:.5f}")
    return mc

print("\n=== обучение на отборочном протоколе ===")
t0 = time.time()
m = fit(SEL_TR)
torch.save(m.state_dict(), OUT / "q2c_sel.pt")
torch.cuda.empty_cache()
print(f"  {time.time()-t0:.0f}s")
print("\n=== подбор sigma на обучающих якорях ===")
sigma = fit_sigma(m, SEL_TR)
print("\n=== оценка ===")
mc0 = report(m, 0.0, SEL_EV, "без frailty")
mc1 = report(m, sigma, SEL_EV, f"frailty {sigma}")
np.save(OUT / "q2c_daily_sel.npy", mc1)

# декорреляция против дерева на том же якоре
import lightgbm as lgb
X, _ = build_features(sorted(set(SEL_TR + [SEL_EV])), verbose=False)
Xtr = np.concatenate([X[a] for a in SEL_TR], 0)
ytr = np.log1p(np.concatenate([target(a) for a in SEL_TR]))
pl_ = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=63,
                        min_child_samples=100, max_bin=63, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=5.0, n_jobs=11, verbose=-1,
                        random_state=42, force_col_wise=True,
                        objective="regression").fit(Xtr, ytr).predict(X[SEL_EV])
del Xtr, X
print(f"\n  LightGBM на том же якоре: {rmsle(target(SEL_EV), np.expm1(pl_)):.5f}")
print(f"  corr(дневная, LightGBM) = {np.corrcoef(mc1, pl_)[0,1]:.5f}")

print("\n=== финальный вектор для лидерборда (якорь 408) ===")
t0 = time.time()
mf = fit(FIN_TR)
torch.save(mf.state_dict(), OUT / "q2c_fin.pt")
torch.cuda.empty_cache()
sig_f = fit_sigma(mf, FIN_TR)
fin, _ = simulate(mf, FIN_EV, sig_f)
fin = fin - fin.mean() + EL
import polars as pl
uid = np.load("cache/uids.npy")
pl.DataFrame({"user_id": uid, "predict": np.clip(np.expm1(fin), 0, None)}) \
  .write_csv("submissions/daily_gen_q2c.csv")
print(f"  submissions/daily_gen.csv  mean log1p {fin.mean():.4f}  ({time.time()-t0:.0f}s)")
print("\ndone")
