"""Диффузионные модели для прогноза: условная диффузия по горизонту против нашей дневной модели.

Разбор двух источников.

TimeDiffusion (timetoai) — неприменим структурно. Философия библиотеки: модель обучается ТОЛЬКО
на исходной последовательности, одна модель на один ряд, как ARIMA. У нас 250 000 рядов по 409
точек с 95% нулей. Закрыто по той же причине, что SARIMA.

TSDiff (amazon-science) — идея применима: безусловная диффузия на корпусе рядов плюс self-guidance
на инференсе. Два преимущества перед Toto: порождает СОВМЕСТНЫЕ пути (E[log1p(Σ)] честно) и
обучается на НАШИХ данных, то есть нулевую массу видит. Но их репозиторий построен вокруг
GluonTS-датасетов, а self-guidance требует градиентов через денойзер на каждом шаге — при 250k
клиентов неподъёмно. Условная диффузия дешевле безусловной с наведением и сэмплово эффективнее,
поэтому здесь она.

ГЛАВНАЯ ПРОВЕРКА — атом в нуле. У гауссовой диффузии та же болезнь, что убила Toto zero-shot:
дельта-функция в нуле размывается, нули становятся малыми положительными, P(Y>0) уезжает к
единице. Наша дневная модель лечит это явным хазард-выходом. Отсюда две руки:

  diff_raw     чистая условная диффузия по 30 дням в log1p-пространстве
  diff_hazard  диффузия описывает УСЛОВНУЮ величину, нули отдельным хазард-выходом

Если diff_hazard заметно лучше diff_raw — подтверждено, что дело в атоме, а не в диффузии.
Если обе проигрывают нашей модели — диффузия не даёт ничего сверх хазарда с логнормалью.

Условие ставится нашим же свёрточным стволом, чтобы сравнение было о ГОЛОВЕ, а не о входе.
"""
import sys, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L = 364
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
EPOCHS, BS, LR = 8, 512, 2e-3
T_DIFF = int(os.environ.get("T_DIFF", 100))
DDIM_STEPS = int(os.environ.get("DDIM_STEPS", 25))
NPATH = int(os.environ.get("NPATH", 64))

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
SEQ = np.stack([(np.log1p(panel(n).astype(np.float32))
                 if panel(n).max() > 3 else panel(n).astype(np.float32)).astype(np.float16)
                for n in CH_NAMES], 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}, шагов диффузии {T_DIFF}, DDIM {DDIM_STEPS}", flush=True)

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

# косинусное расписание шума
tt = torch.linspace(0, 1, T_DIFF + 1)
ab = torch.cos((tt + 0.008) / 1.008 * math.pi / 2) ** 2
ALPHA_BAR = (ab / ab[0]).to(DEV)

class Trunk(nn.Module):
    """тот же свёрточный ствол, что у нашей дневной модели — сравнение о голове, не о входе"""
    def __init__(self, c, h=128, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.out = h * 3
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return torch.cat([z.mean(-1), z.max(-1).values, z[:, :, -30:].mean(-1)], -1)

def temb(t, d=128):
    half = d // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)

class Denoiser(nn.Module):
    """предсказывает шум по зашумлённому горизонту, условию и номеру шага"""
    def __init__(self, cond_dim, d=256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(cond_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.tproj = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.inp = nn.Conv1d(1, 64, 5, padding=2)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(64 + d, 64, 3, padding=p, dilation=p), nn.GroupNorm(8, 64), nn.SiLU())
            for p in (1, 2, 4, 8)])
        self.out = nn.Conv1d(64, 1, 3, padding=1)
    def forward(self, x, c, t):
        h = self.inp(x.unsqueeze(1))
        ctx = (self.cond(c) + self.tproj(temb(t))).unsqueeze(-1).expand(-1, -1, x.shape[1])
        for b in self.bl:
            h = h + b(torch.cat([h, ctx], 1))
        return self.out(h).squeeze(1)

class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.trunk = Trunk(CH)
        self.den = Denoiser(self.trunk.out)
        self.gate = nn.Sequential(nn.Linear(self.trunk.out, 256), nn.SiLU(),
                                  nn.Linear(256, HOR)) if arm == "diff_hazard" else None

def q_sample(x0, t, noise):
    a = ALPHA_BAR[t][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise

def train(arm, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Model(arm).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    for ep in range(EPOCHS):
        m.train(); run = nb = 0
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64: continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
                ly = torch.log1p(yd)
                c = m.trunk(x)
                pos = (yd > 0).float()
                if arm == "diff_hazard":
                    # диффузия описывает только УСЛОВНУЮ величину; нули берёт на себя гейт
                    x0 = ly
                    gl = Fn.binary_cross_entropy_with_logits(m.gate(c), pos, reduction="none").mean()
                else:
                    x0 = ly
                    gl = 0.0
                t = torch.randint(1, T_DIFF + 1, (len(idx),), device=DEV)
                noise = torch.randn_like(x0)
                xt = q_sample(x0, t, noise)
                pred = m.den(xt, c, t)
                w = pos if arm == "diff_hazard" else torch.ones_like(pos)
                dl = ((pred - noise) ** 2 * w).sum() / w.sum().clamp(min=1)
                loss = dl + gl
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
                run += float(loss); nb += 1
        print(f"    [{arm}] эпоха {ep}: loss {run/max(nb,1):.4f}", flush=True)
    return m.eval()

@torch.no_grad()
def sample(m, anchor, chunk=512):
    """DDIM: детерминированная траектория, шум добавляется только через стартовую точку"""
    ts = torch.linspace(T_DIFF, 1, DDIM_STEPS, device=DEV).long()
    out = np.empty(NU, np.float32)
    for i in range(0, NU, chunk):
        idx = np.arange(i, min(i + chunk, NU))
        b = len(idx)
        c = m.trunk(torch.from_numpy(win(idx, anchor)).to(DEV))
        cR = c.repeat_interleave(NPATH, 0)
        x = torch.randn(b * NPATH, HOR, device=DEV)
        for k in range(len(ts)):
            t = ts[k].expand(b * NPATH)
            eps = m.den(x, cR, t)
            a_t = ALPHA_BAR[t][:, None]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-4)
            x0 = x0.clamp(0, 13)
            a_p = ALPHA_BAR[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=DEV)
            x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
        val = torch.expm1(x0.clamp(0, 13))
        if m.arm == "diff_hazard":
            p = torch.sigmoid(m.gate(c)).repeat_interleave(NPATH, 0)
            val = val * (torch.rand_like(p) < p).float()
        tot = val.clamp(min=0).sum(-1).view(b, NPATH)
        out[idx] = torch.log1p(tot).mean(-1).float().cpu().numpy()
    return out

y = target(SEL_EV); ly_ = np.log1p(y)
ref = OUT / "h8_gauss.npy"
if ref.exists():
    r = np.load(ref); r = r - r.mean() + ly_.mean()
    print(f"\nнаша дневная модель: {rmsle(y, np.expm1(np.clip(r,0,13))):.5f}", flush=True)
print(f"эмпирическая P(Y>0) = {(y>0).mean():.4f}\n", flush=True)

print(f"{'рука':>13} {'RMSLE':>9} {'P(Y>0)':>9} {'сек':>7}", flush=True)
RES = {}
for arm in ("diff_raw", "diff_hazard"):
    t0 = time.time()
    try:
        m = train(arm)
        v = sample(m, SEL_EV)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"{arm:>13}  ОШИБКА {type(e).__name__}: {str(e)[:110]}", flush=True)
        continue
    RES[arm] = v
    sh = v - v.mean() + ly_.mean()
    pos_rate = float((np.expm1(np.clip(v, 0, 13)) > 0.5).mean())
    print(f"{arm:>13} {rmsle(y, np.expm1(np.clip(sh,0,13))):9.5f} {pos_rate:9.4f} "
          f"{time.time()-t0:7.0f}", flush=True)
    np.save(OUT / f"m1_{arm}.npy", v)
if len(RES) == 2:
    print(f"\ncorr между руками: {np.corrcoef(RES['diff_raw'],RES['diff_hazard'])[0,1]:.5f}")
if ref.exists() and RES:
    for k, v in RES.items():
        print(f"corr({k}, наша дневная) = {np.corrcoef(v, np.load(ref))[0,1]:.5f}")
print("\nчитать так: diff_hazard заметно лучше diff_raw — дело в атоме нуля, а не в диффузии;")
print("обе хуже нашей модели — диффузия не даёт ничего сверх хазарда с логнормалью")
print("\ndone")
