"""Дообучение moirai-1.1-R-large на наших данных — полное, не LoRA.

Обучаем ИХ собственным объективом: NLL смеси (StudentT + NormalFixedScale + NegativeBinomial +
LogNormal) на предсказываемых токенах. Это честное дообучение foundation-модели на нашем домене,
а не подмена задачи нашей головой.

Вход собирает сама библиотека: fc._convert(patch, past_target, past_observed, past_is_pad,
future_target, future_observed) отдаёт шесть тензоров, седьмым добавляется patch_size по токенам.
Угадывать раскладку масок не нужно — это тот же принцип, что спас нас на PatchTST-FM.

Прецедент против: у Toto полное дообучение вышло ХУЖЕ заморозки (1,77137 против 1,76758).
Априори здесь тот же исход вероятнее. Смысл проверки в том, что у Moirai другое правдоподобие
(со счётной компонентой) и другой объектив, поэтому перенос прецедента не автоматический.

После дообучения — те же три замера, что в t1, чтобы числа были сопоставимы:
zero-shot, замороженный ствол с нашей головой, и контроль случайной проекции.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from uni2ts.model.moirai import MoiraiModule, MoiraiForecast

DEV = "cuda"
CTX, PATCH = 320, 32
NPATCH, PDIM = 11, 256
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS = int(os.environ.get("BS", 96))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", 1))
FT_LR = float(os.environ.get("FT_LR", 1e-5))
NSAMP_ZS = 100
EPOCHS, BSH, LR = 8, 512, 2e-3
NPATH, GH, SIGMA = 96, 16, 0.8
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
CHS = [G, V, C]

def win1(M, users, end):
    lo = end - CTX + 1
    if lo >= 0:
        return M[users, lo:end + 1].astype(np.float32)
    got = M[users, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), -lo), np.float32), got], 1)

y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))

print("загружаю moirai-1.1-R-large ...", flush=True)
mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large").to(DEV)
fc = MoiraiForecast(prediction_length=HOR, target_dim=1, feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0, context_length=CTX, module=mod,
                    patch_size=PATCH, num_samples=NSAMP_ZS).to(DEV)
print(f"  {sum(p.numel() for p in mod.parameters())/1e6:.1f}М параметров, "
      f"дообучение {FT_EPOCHS} эпох при шаге {FT_LR}", flush=True)

def batch_inputs(idx, a, with_future=True):
    pt = torch.from_numpy(win1(G, idx, a)).to(DEV).unsqueeze(-1)
    po = torch.ones_like(pt, dtype=torch.bool)
    pp = torch.zeros(len(idx), CTX, dtype=torch.bool, device=DEV)
    if not with_future:
        return fc._convert(PATCH, pt, po, pp)
    ft = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV).unsqueeze(-1)
    fo = torch.ones_like(ft, dtype=torch.bool)
    return fc._convert(PATCH, pt, po, pp, future_target=ft, future_observed_target=fo)

# --------------------------- дообучение ---------------------------
opt = torch.optim.AdamW(mod.parameters(), lr=FT_LR, weight_decay=1e-2)
steps = FT_EPOCHS * len(SEL_TR) * (NU // BS)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=FT_LR, total_steps=max(steps, 10),
                                          pct_start=0.05)
rng = np.random.default_rng(0)
print(f"\nдообучение: {steps} шагов", flush=True)
t0 = time.time(); mod.train()
for ep in range(FT_EPOCHS):
    tot, nb = 0.0, 0
    for a in SEL_TR:
        perm = rng.permutation(NU)
        for i in range(0, NU, BS):
            idx = np.sort(perm[i:i + BS])
            if len(idx) < 16: continue
            out = list(batch_inputs(idx, a))
            ps = torch.full(out[2].shape, PATCH, dtype=torch.long, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = mod(*out, ps)
                lp = d.log_prob(out[0])
                w = (out[5].unsqueeze(-1) & out[1]).float()
                loss = -(lp * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            if sch.last_epoch < steps - 1: sch.step()
            tot += float(loss); nb += 1
            if nb % 500 == 0:
                print(f"    шаг {nb}: NLL {tot/nb:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  эпоха {ep}: NLL {tot/max(nb,1):.4f}  ({time.time()-t0:.0f}s)", flush=True)
mod.eval()

print(f"\nориентиры: наша свёртка 1,73808 | Moirai zero-shot ДО дообучения 1,88394 | "
      f"ствол Toto 313m 1,76758 при контроле 1,77794", flush=True)

# --------------------------- zero-shot после дообучения ---------------------------
t0 = time.time(); zs = np.empty(NU, np.float32)
with torch.no_grad():
    for i in range(0, NU, BS):
        u = np.arange(i, min(i + BS, NU))
        pt = torch.from_numpy(win1(G, u, SEL_EV)).to(DEV).unsqueeze(-1)
        po = torch.ones_like(pt, dtype=torch.bool)
        pp = torch.zeros(len(u), CTX, dtype=torch.bool, device=DEV)
        s = fc(past_target=pt, past_observed_target=po, past_is_pad=pp)
        zs[u] = torch.log1p(s.clamp(min=0).sum(-1)).mean(-1).float().cpu().numpy()
print(f"\n{'рука':>16} {'RMSLE':>10} {'сек':>7}", flush=True)
print(f"{'zs_после_ft':>16} {sc(zs.astype(np.float64)):10.5f} {time.time()-t0:7.0f}", flush=True)
np.save(OUT / "t2_moirai_ft_zs.npy", zs)

# --------------------------- ствол после дообучения ---------------------------
BUF = {}
mod.param_proj.register_forward_pre_hook(lambda m, i: BUF.__setitem__("h", i[0].detach()))
gen = torch.Generator(device="cpu").manual_seed(0)
PROJ = (torch.randn(1024, PDIM, generator=gen) / np.sqrt(1024)).to(DEV)

def embed(anchor):
    outs = []
    for i in range(0, NU, BS):
        u = np.arange(i, min(i + BS, NU))
        pt = torch.from_numpy(win1(G, u, anchor)).to(DEV).unsqueeze(-1)
        po = torch.ones_like(pt, dtype=torch.bool)
        pp = torch.zeros(len(u), CTX, dtype=torch.bool, device=DEV)
        with torch.no_grad():
            fc(past_target=pt, past_observed_target=po, past_is_pad=pp)
        h = BUF["h"]
        if i == 0: assert h.shape[1] == NPATCH, h.shape
        outs.append((h @ PROJ).reshape(len(u), -1).half().cpu().numpy())
    return np.concatenate(outs, 0)

print("\nснимаю представления дообученного ствола ...", flush=True)
t0 = time.time(); EMB = {}
for a in SEL_TR + [SEL_EV]:
    EMB[a] = embed(a)
    print(f"  якорь {a}: {EMB[a].shape} ({time.time()-t0:.0f}s)", flush=True)
D = EMB[SEL_EV].shape[1]
del fc, mod
torch.cuda.empty_cache()

def normalize(F):
    mu = np.mean([F[a].astype(np.float32).mean(0) for a in SEL_TR], 0)
    sd = np.maximum(np.mean([F[a].astype(np.float32).std(0) for a in SEL_TR], 0), 1e-3)
    for a in F:
        F[a] = np.clip((F[a].astype(np.float32) - mu) / sd, -8, 8).astype(np.float16)
    return F
EMB = normalize(EMB)

class Head(nn.Module):
    def __init__(self, din, npatch, d=192):
        super().__init__()
        self.np_, self.pd = npatch, din // npatch
        self.q = nn.Parameter(torch.randn(1, 1, self.pd) * 0.02)
        self.pool = nn.MultiheadAttention(self.pd, 4, batch_first=True)
        self.body = nn.Sequential(nn.Linear(self.pd * 2, 512), nn.SiLU(), nn.Linear(512, d), nn.SiLU())
        self.pos = nn.Parameter(torch.randn(HOR, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(d + 16, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, e):
        B = e.shape[0]
        p = e.view(B, self.np_, self.pd)
        a = self.pool(self.q.expand(B, 1, -1), p, p, need_weights=False)[0].squeeze(1)
        h = self.body(torch.cat([a, p[:, -1]], -1))
        z = torch.cat([h.unsqueeze(1).expand(B, HOR, -1), self.pos.unsqueeze(0).expand(B, HOR, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); lyd = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyd - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def run_head(feats, seed=0):
    torch.manual_seed(seed); r2 = np.random.default_rng(seed)
    m = Head(feats[SEL_EV].shape[1], NPATCH).to(DEV)
    o2 = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    st = EPOCHS * len(SEL_TR) * (NU // BSH)
    s2 = torch.optim.lr_scheduler.OneCycleLR(o2, max_lr=LR, total_steps=max(st, 10))
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = r2.permutation(NU)
            for i in range(0, NU, BSH):
                idx = np.sort(perm[i:i + BSH])
                if len(idx) < 64: continue
                e = torch.from_numpy(feats[a][idx].astype(np.float32)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
                o2.zero_grad(set_to_none=True)
                loss_fn(*m(e), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); o2.step()
                if s2.last_epoch < st - 1: s2.step()
    m.eval()
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 1024):
            idx = np.arange(i, min(i + 1024, NU))
            hz, mu, ls = m(torch.from_numpy(feats[SEL_EV][idx].astype(np.float32)).to(DEV))
            hz = hz * scale; sd = ls.exp()
            acc = torch.zeros(len(idx), device=DEV)
            for zn, wn in zip(nodes, wts):
                p = torch.sigmoid(hz + float(zn) * SIGMA)
                u = torch.rand(NPATH, len(idx), HOR, device=DEV) < p.unsqueeze(0)
                v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                * torch.randn(NPATH, len(idx), HOR, device=DEV))
                acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
            out[idx] = acc.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out

t0 = time.time(); vt = run_head(EMB)
print(f"{'trunk_после_ft':>16} {sc(vt.astype(np.float64)):10.5f} {time.time()-t0:7.0f}", flush=True)
np.save(OUT / "t2_moirai_ft_trunk.npy", vt)

rs = np.random.default_rng(0)
W = (rs.standard_normal((len(CHS) * CTX, D)) / np.sqrt(len(CHS) * CTX)).astype(np.float32)
RND = {}
for a in SEL_TR + [SEL_EV]:
    X = np.log1p(np.maximum(np.concatenate([win1(M, np.arange(NU), a) for M in CHS], 1), 0))
    RND[a] = (X @ W).astype(np.float16); del X
RND = normalize(RND)
t0 = time.time(); vc = run_head(RND)
print(f"{'control':>16} {sc(vc.astype(np.float64)):10.5f} {time.time()-t0:7.0f}", flush=True)
print(f"\nотрыв дообученного ствола от контроля: "
      f"{sc(vt.astype(np.float64)) - sc(vc.astype(np.float64)):+.5f}")
print("  (Toto 313m заморозка −0,01036, её же полное дообучение вышло ХУЖЕ заморозки)")
print("\ndone")
