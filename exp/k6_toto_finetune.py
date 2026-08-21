"""Полное дообучение ствола Toto 2.0-313m на нашу задачу.

Основание. Замороженный ствол несёт реальный сигнал: отрыв от случайной проекции 0.0104 на 313m
и 0.0113 на 1B, устойчиво. Но потолок 1.766 против нашей свёртки 1.73808, и разрыв не берётся ни
отводом (расширение учетверило отрыв, не сдвинув скор), ни масштабом (22m->313m дал 0.0026,
313m->1B только 0.0010).

Плоская кривая размера означает, что упирается не ёмкость, а ДОМЕН. Zero-shot это показал в чистом
виде: P(Y>0) = 0.9967 против эмпирических 0.5830 — модель обучена на телеметрии, где нулей нет.
Дообучение — единственный рычаг, который до нуля ещё не доводили.

КОНТРОЛЬ ОБЯЗАТЕЛЕН. Та же архитектура со СЛУЧАЙНОЙ инициализацией, обученная точно так же.
Без него нельзя отличить «предобучение помогло» от «трансформер на 313М, обученный на наших
данных, и так работает». Запускается второй рукой, если первая покажет смысл.

Против дообучения есть прецедент: в e3 предобучение на недельном горизонте (на НАШИХ же данных)
дало +0.00200, а pre_frozen совпал с pre_ft до пятого знака — дообучение не сдвинуло ствол из ямы
предобучения. Здесь домен чужой сильнее, но и разрыв больше.

Разные скорости обучения: ствол 1e-5, голова 1e-3 — стандарт для дообучения предобученного
трансформера. Оценка после каждой эпохи, чтобы видеть траекторию.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, H as HOR, OUT
from toto2 import Toto2Model

DEV = "cuda"
CTX, NPATCH = 320, 10
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348
BS = int(os.environ.get("BS", 48))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR_TRUNK, LR_HEAD = 1e-5, 1e-3
NSAMP, GH, SIGMA = 96, 16, 0.8
MODEL = os.environ.get("TOTO_MODEL", "Datadog/Toto-2.0-313m")
ARM = os.environ.get("ARM", "pretrained")     # pretrained | random_init
G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
CH = [G, V, C]

def win(users, end):
    lo = end - CTX + 1
    if lo >= 0:
        return np.stack([M[users, lo:end + 1] for M in CH], 1).astype(np.float32)
    got = np.stack([M[users, 0:end + 1] for M in CH], 1).astype(np.float32)
    return np.concatenate([np.zeros((len(users), len(CH), -lo), np.float32), got], 2)

class Net(nn.Module):
    def __init__(self, arm, d=192):
        super().__init__()
        self.toto = Toto2Model.from_pretrained(MODEL)
        if arm == "random_init":
            for m in self.toto.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        self.cap = {}
        self.toto.transformer.register_forward_hook(
            lambda _m, _i, o: self.cap.__setitem__("h", o[0] if isinstance(o, (tuple, list)) else o))
        pd = 1024
        self.q = nn.Parameter(torch.randn(1, 1, pd) * 0.02)
        self.pool = nn.MultiheadAttention(pd, 8, batch_first=True)
        self.body = nn.Sequential(nn.Linear(pd * 2, 512), nn.SiLU(), nn.Linear(512, d), nn.SiLU())
        self.pos = nn.Parameter(torch.randn(HOR, 16) * 0.02)
        self.net = nn.Sequential(nn.Linear(d + 16, d), nn.SiLU(), nn.Linear(d, 3))
    def forward(self, x):
        msk = torch.ones_like(x, dtype=torch.bool)
        sid = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
        self.toto(target=x, target_mask=msk, cpm_mask=torch.ones_like(msk), series_ids=sid)
        h = self.cap["h"]
        p = h[:, 0] if h.dim() == 4 else h.unsqueeze(1)
        B = p.shape[0]
        a = self.pool(self.q.expand(B, 1, -1), p, p, need_weights=False)[0].squeeze(1)
        z0 = self.body(torch.cat([a, p[:, -1]], -1))
        z = torch.cat([z0.unsqueeze(1).expand(B, HOR, -1),
                       self.pos.unsqueeze(0).expand(B, HOR, -1)], -1)
        o = self.net(z)
        return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)

def loss_fn(hz, mu, ls, yd):
    pos = (yd > 0).float(); lyd = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((lyd - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

@torch.no_grad()
def evaluate(m):
    m.eval()
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32)
    for i in range(0, NU, 256):
        idx = np.arange(i, min(i + 256, NU))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hz, mu, ls = m(torch.from_numpy(win(idx, SEL_EV)).to(DEV))
        hz, mu, ls = hz.float(), mu.float(), ls.float()
        hz = hz * scale; sd = ls.exp()
        acc = torch.zeros(len(idx), device=DEV)
        for zn, wn in zip(nodes, wts):
            p = torch.sigmoid(hz + float(zn) * SIGMA)
            u = torch.rand(NSAMP, len(idx), HOR, device=DEV) < p.unsqueeze(0)
            v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                            * torch.randn(NSAMP, len(idx), HOR, device=DEV))
            acc += float(wn) * torch.log1p((u.float() * v.clamp(min=0)).sum(-1)).mean(0)
        out[idx] = acc.float().cpu().numpy()
    m.train()
    return out

y = target(SEL_EV); ly = np.log1p(y)
print(f"рука {ARM}, модель {MODEL}, батч {BS}, эпох {EPOCHS}", flush=True)
print(f"ориентиры: наша свёртка 1.73808, замороженный ствол 313m 1.76758", flush=True)

torch.manual_seed(0)
m = Net(ARM).to(DEV)
ntr = sum(p.numel() for p in m.toto.parameters()) / 1e6
nhd = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("toto")) / 1e6
print(f"ствол {ntr:.0f}М (обучается), голова {nhd:.2f}М", flush=True)

opt = torch.optim.AdamW([
    {"params": m.toto.parameters(), "lr": LR_TRUNK},
    {"params": [p for n, p in m.named_parameters() if not n.startswith("toto")], "lr": LR_HEAD},
], weight_decay=1e-5)
steps = EPOCHS * len(SEL_TR) * (NU // BS)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_TRUNK, LR_HEAD], total_steps=max(steps, 10))
scaler_amp = torch.amp.GradScaler("cuda")
rng = np.random.default_rng(0)
best = None
for ep in range(EPOCHS):
    t0 = time.time(); run = nb = 0
    m.train()
    for a in SEL_TR:
        perm = rng.permutation(NU)
        for i in range(0, NU, BS):
            idx = np.sort(perm[i:i + BS])
            if len(idx) < 16: continue
            x = torch.from_numpy(win(idx, a)).to(DEV)
            yd = torch.from_numpy(G[idx, a + 1:a + 1 + HOR].astype(np.float32)).to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(x)
            l = loss_fn(o[0].float(), o[1].float(), o[2].float(), yd)
            l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if sch.last_epoch < steps - 1: sch.step()
            run += l.item(); nb += 1
    v = evaluate(m)
    sh = v - v.mean() + ly.mean()
    r = rmsle(y, np.expm1(np.clip(sh, 0, 13)))
    print(f"  эпоха {ep}: loss {run/max(nb,1):.4f}  RMSLE {r:.5f}  ({time.time()-t0:.0f}s)",
          flush=True)
    if best is None or r < best[0]:
        best = (r, v.copy())
        np.save(OUT / f"k6_{ARM}.npy", v)
print(f"\nлучшее у {ARM}: {best[0]:.5f}")
ref = OUT / "h8_gauss.npy"
if ref.exists():
    print(f"corr с нашей дневной: {np.corrcoef(best[1], np.load(ref))[0,1]:.5f}")
print("done")
