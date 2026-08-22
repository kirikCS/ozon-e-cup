"""Moirai-1.1-R-large: единственная TSFM со счётным правдоподобием в смеси.

Почему именно она, а не Moirai 2.0 и не агент. Смесь выходного распределения здесь —
StudentT + NormalFixedScale + **NegativeBinomial** + LogNormal, и NB это единственная счётная
компонента во всём ландшафте TSFM. Наш центральный диагноз весь проект в том, что непрерывная
плотность не может иметь атома в нуле (у нас его 43,7%), и NB — единственное правдоподобие,
которое дискретность несёт честно. В Moirai 2.0 эту компоненту ВЫКИНУЛИ при переходе на pinball.
311М параметров проходят наш измеренный порог: Toto развернулась между 120М и 313М.

Второе преимущество, вскрытое при разведке API: MoiraiForecast отдаёт СЭМПЛЫ траекторий
(batch, num_samples, horizon), а не квантили. Значит E[log1p(sum y_d)] считается напрямую
усреднением log1p суммы по сэмплам — тот самый функционал, который оптимален под RMSLE.
Именно его невозможно было корректно извлечь из квантильных выходов Chronos-2 и Toto, и
именно это их убило (1,883 и 1,958 против наших 1,738).

Три руки, как во всех прошлых проверках foundation-моделей:
  zero_shot — прогноз как есть, считывание средним log1p суммы по сэмплам;
  trunk     — замороженный энкодер + НАША генеративная голова (хазард + логнормаль + frailty);
  control   — случайная линейная проекция сырого окна ТОЙ ЖЕ размерности через ту же голову.

Контроль обязателен: у Toto он подтвердил сигнал, у Chronos-2 снял его, у PatchTST-FM показал
отрыв всего 0,001. Без него «1,77» читается как «сопоставимо с базой», а не как «хуже случайной
проекции».
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
NPATCH, PDIM = 11, 256                 # 11 токенов x 256 = 2816, как у Chronos (2816) и Toto (2560)
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS = int(os.environ.get("BS", 128))
NSAMP_ZS = int(os.environ.get("NSAMP_ZS", 100))
EPOCHS, BSH, LR = 8, 512, 2e-3
NPATH, GH, SIGMA = 96, 16, 0.8
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
CH = [G, V, C]

def win1(M, users, end):
    lo = end - CTX + 1
    if lo >= 0:
        return M[users, lo:end + 1].astype(np.float32)
    got = M[users, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), -lo), np.float32), got], 1)

print("загружаю moirai-1.1-R-large ...", flush=True)
mod = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large")
print(f"  {sum(p.numel() for p in mod.parameters())/1e6:.1f}М параметров", flush=True)
comps = [type(c).__name__ for c in mod.distr_output.components]
print(f"  смесь: {comps}", flush=True)
assert any("NegativeBinomial" in c for c in comps), "NB компонента пропала — не та модель"

y = target(SEL_EV); ly = np.log1p(y); MU = ly.mean()
def sc(v): return rmsle(y, np.expm1(np.clip(v - v.mean() + MU, 0, 13)))
ref = OUT / "h8_gauss.npy"
if ref.exists():
    r = np.load(ref)
    print(f"\nориентиры: наша свёртка {sc(r.astype(np.float64)):.5f}; ствол Toto 313m 1,76758 "
          f"при контроле 1,77794; Chronos-2 1,77785 при контроле 1,77719", flush=True)

# --------------------------- рука 1: zero-shot ---------------------------
fc = MoiraiForecast(prediction_length=HOR, target_dim=1, feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0, context_length=CTX, module=mod,
                    patch_size=PATCH, num_samples=NSAMP_ZS).to(DEV).eval()
print(f"\nzero-shot, {NSAMP_ZS} сэмплов ...", flush=True)
t0 = time.time(); zs = np.empty(NU, np.float32)
with torch.no_grad():
    for i in range(0, NU, BS):
        u = np.arange(i, min(i + BS, NU))
        pt = torch.from_numpy(win1(G, u, SEL_EV)).to(DEV).unsqueeze(-1)
        po = torch.ones_like(pt, dtype=torch.bool)
        pp = torch.zeros(len(u), CTX, dtype=torch.bool, device=DEV)
        s = fc(past_target=pt, past_observed_target=po, past_is_pad=pp)   # (b, S, HOR)
        # ТОЧНЫЙ функционал: среднее log1p суммы по сэмплам, а не log1p среднего
        zs[u] = torch.log1p(s.clamp(min=0).sum(-1)).mean(-1).float().cpu().numpy()
        if i == 0:
            print(f"    форма выхода {tuple(s.shape)}", flush=True)
print(f"{'zero_shot':>12} {sc(zs.astype(np.float64)):10.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(OUT / "t1_moirai_zs.npy", zs)
print(f"  P(Y>0) у модели {np.mean(zs > 1e-3):.4f} против факта {(y > 0).mean():.4f}", flush=True)

# --------------------------- отвод со ствола ---------------------------
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
        if i == 0:
            print(f"    представление {tuple(h.shape)}", flush=True)
            assert h.shape[1] == NPATCH, h.shape
        outs.append((h @ PROJ).reshape(len(u), -1).half().cpu().numpy())
    return np.concatenate(outs, 0)

print("\nснимаю представления ...", flush=True)
t0 = time.time(); EMB = {}
for a in SEL_TR + [SEL_EV]:
    EMB[a] = embed(a)
    print(f"  якорь {a}: {EMB[a].shape}  {EMB[a].nbytes/1e9:.2f} ГБ  ({time.time()-t0:.0f}s)", flush=True)
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
    """attention-пуллинг по патчам + последний патч, затем наша генеративная голова"""
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
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Head(feats[SEL_EV].shape[1], NPATCH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BSH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BSH):
                idx = np.sort(perm[i:i + BSH])
                if len(idx) < 64: continue
                e = torch.from_numpy(feats[a][idx].astype(np.float32)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                loss_fn(*m(e), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
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

print(f"\n{'рука':>12} {'признаков':>10} {'RMSLE':>10} {'сек':>7}", flush=True)
t0 = time.time(); vt = run_head(EMB)
print(f"{'trunk':>12} {D:10d} {sc(vt.astype(np.float64)):10.5f} {time.time()-t0:7.0f}", flush=True)
np.save(OUT / "t1_moirai_trunk.npy", vt)

rs = np.random.default_rng(0)
W = (rs.standard_normal((len(CH) * CTX, D)) / np.sqrt(len(CH) * CTX)).astype(np.float32)
RND = {}
for a in SEL_TR + [SEL_EV]:
    X = np.log1p(np.maximum(np.concatenate([win1(M, np.arange(NU), a) for M in CH], 1), 0))
    RND[a] = (X @ W).astype(np.float16); del X
RND = normalize(RND)
t0 = time.time(); vc = run_head(RND)
print(f"{'control':>12} {D:10d} {sc(vc.astype(np.float64)):10.5f} {time.time()-t0:7.0f}", flush=True)
np.save(OUT / "t1_moirai_ctrl.npy", vc)
print(f"\ncorr(trunk, control) = {np.corrcoef(vt, vc)[0,1]:.5f}")
if ref.exists():
    print(f"corr(trunk, наша дневная) = {np.corrcoef(vt, np.load(ref))[0,1]:.5f}")
print(f"\nотрыв ствола от контроля: {sc(vt.astype(np.float64)) - sc(vc.astype(np.float64)):+.5f}")
print("  (Toto 313m: −0,01036 | Chronos-2: +0,00066 | PatchTST-FM: −0,00100)")
print("\ndone")
