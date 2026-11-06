"""Квантильные руки на финальном якоре — продакшн-версия для стека.

p6 проверил допуск напрямую: гребневая смесь двадцати наших векторов на якоре 348, веса на
половине клиентов, скор на другой. Генеративные руки не дали ничего (+2,2e-5 и -1,7e-5), а
квантильные -2,61e-4 и -3,66e-4 при контроле на перемешанном кандидате +2,2e-5 и +1,0e-5,
то есть выигрыш в тридцать раз больше цены лишнего столбца. Обе вместе -3,95e-4, дополняют
друг друга. p7 показал, что вклад тает с ростом базиса медленно: -1,07e-3 при трёх векторах,
-3,95e-4 при двадцати.

Голова проигрывает по скору на всех шести проверенных стволах и всё равно полезна стеку -
случай event_multi, который вошёл с худшим локальным скором и получил наибольший вес.

Здесь обучение на FIN_TR с предсказанием на 408. Цели за 408 не существует (панель 409 дней),
поэтому скора нет - только векторы. Два ствола, два сида на каждый, усреднение: бэггинг у
дневной модели давал -0,00252.
"""
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, H, D0, ds, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, NPATH, GH = 364, 8, 512, 96, 16
PLEN, WLEN = 14, 7
SEL_TR, SEL_EV = [378, 350, 322, 294, 266], 408   # ФИНАЛЬНЫЙ протокол
SIGMA = 0.8

CH_NAMES = ["gmv", "gmv_search", "ord", "cart", "srch", "visit", "engaged",
            "scart", "sord", "sday", "catday"]
chans = []
for n in CH_NAMES:
    P = panel(n).astype(np.float32)
    chans.append((np.log1p(P) if P.max() > 3 else P).astype(np.float16))
SEQ = np.stack(chans, 1)
CH = SEQ.shape[1]
G = panel("gmv")
print(f"SEQ {SEQ.shape}, устройство {DEV}", flush=True)

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

class Block(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * d, d))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))

class CNNTrunk(nn.Module):
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

class PatchTrunk(nn.Module):
    """attention over patches; plen=14 gives 26 tokens, plen=7 with weekly sums gives 52"""
    def __init__(self, c, plen, d=128, nl=3, weekly=False):
        super().__init__()
        self.plen, self.weekly, self.n = plen, weekly, L // plen
        self.emb = nn.Linear(c if weekly else c * plen, d)
        self.pos = nn.Parameter(torch.randn(1, self.n, d) * 0.02)
        self.bl = nn.ModuleList([Block(d) for _ in range(nl)])
        self.out = d * 2
    def forward(self, x):
        B = x.shape[0]
        p = x.reshape(B, CH, self.n, self.plen)
        p = p.sum(-1) .transpose(1, 2) if self.weekly else \
            p.permute(0, 2, 1, 3).reshape(B, self.n, CH * self.plen)
        z = self.emb(p) + self.pos
        for b in self.bl:
            z = b(z)
        return torch.cat([z.mean(1), z[:, -1]], -1)


NQ = 99
TAU = torch.arange(1, NQ + 1, dtype=torch.float32, device=DEV) / (NQ + 1)

class Model(nn.Module):
    def __init__(self, arm, head="gen", d=128):
        super().__init__()
        self.head = head
        if arm == "cnn":
            self.tr = nn.ModuleList([CNNTrunk(CH)])
        elif arm == "senc":
            self.tr = nn.ModuleList([PatchTrunk(CH, PLEN)])
        elif arm == "wenc":
            self.tr = nn.ModuleList([PatchTrunk(CH, WLEN, weekly=True)])
        else:
            self.tr = nn.ModuleList([CNNTrunk(CH), PatchTrunk(CH, PLEN),
                                     PatchTrunk(CH, WLEN, weekly=True)])
        tot = sum(t.out for t in self.tr)
        self.mix = nn.Sequential(nn.Linear(tot, 256), nn.SiLU(), nn.Linear(256, 192)) \
            if len(self.tr) > 1 else nn.Identity()
        hin = 192 if len(self.tr) > 1 else tot
        self.pos = nn.Parameter(torch.randn(H, 16) * 0.02)
        if head == "gen":
            self.net = nn.Sequential(nn.Linear(hin + 10 + 16, d), nn.SiLU(),
                                     nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
        else:
            self.net = nn.Sequential(nn.Linear(hin + 10, d), nn.SiLU(),
                                     nn.Linear(d, d), nn.SiLU(), nn.Linear(d, NQ + 1))
    def forward(self, x, cal):
        h = self.mix(torch.cat([t(x) for t in self.tr], -1))
        B = h.shape[0]
        if self.head == "gen":
            z = torch.cat([h.unsqueeze(1).expand(B, H, -1),
                           cal.unsqueeze(0).expand(B, H, -1),
                           self.pos.unsqueeze(0).expand(B, H, -1)], -1)
            o = self.net(z)
            return o[..., 0], o[..., 1], o[..., 2].clamp(-4, 3)
        o = self.net(torch.cat([h, cal.mean(0).unsqueeze(0).expand(B, -1)], -1))
        return o[:, :1] + torch.cumsum(Fn.softplus(o[:, 1:]) / NQ, 1)

def gen_loss(out, yd):
    hz, mu, ls = out
    pos = (yd > 0).float(); ly = torch.log1p(yd)
    return (Fn.binary_cross_entropy_with_logits(hz, pos, reduction="none")
            + pos * 0.5 * (((ly - mu) / ls.exp()) ** 2 + 2 * ls)).sum(1).mean()

def q_loss(q, yd):
    d = torch.log1p(yd.sum(1)).unsqueeze(1) - q
    return torch.maximum(TAU * d, (TAU - 1) * d).mean(1).mean()

def run(arm, head="gen", seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Model(arm, head).to(DEV)
    lossf = gen_loss if head == "gen" else q_loss
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    steps = EPOCHS * len(SEL_TR) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10))
    CAL = {a: torch.from_numpy(cal_feats(a)).to(DEV) for a in SEL_TR}
    for _ in range(EPOCHS):
        m.train()
        for a in SEL_TR:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64: continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                yd = torch.from_numpy(G[idx, a + 1:a + 1 + H].astype(np.float32)).to(DEV)
                opt.zero_grad(set_to_none=True)
                lossf(m(x, CAL[a]), yd).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
                if sch.last_epoch < steps - 1: sch.step()
    m.eval()
    cal = torch.from_numpy(cal_feats(SEL_EV)).to(DEV)
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH); wts = wts / wts.sum()
    scale = float(np.sqrt(1 + np.pi * SIGMA ** 2 / 8))
    out = np.empty(NU, np.float32); zf = np.empty(NU, np.float32)
    with torch.no_grad():
        for i in range(0, NU, 256):
            idx = np.arange(i, min(i + 256, NU))
            o = m(torch.from_numpy(win(idx, SEL_EV)).to(DEV), cal)
            if head == "gen":
                hz, mu, ls = o
                hz = hz * scale; sd = ls.exp()
                acc = torch.zeros(len(idx), device=DEV)
                for zn, wn in zip(nodes, wts):
                    p = torch.sigmoid(hz + float(zn) * SIGMA)
                    u = torch.rand(NPATH, len(idx), H, device=DEV) < p.unsqueeze(0)
                    v = torch.expm1(mu.unsqueeze(0) + sd.unsqueeze(0)
                                    * torch.randn(NPATH, len(idx), H, device=DEV))
                    acc += float(wn) * torch.log1p((u.float()*v.clamp(min=0)).sum(-1)).mean(0)
                out[idx] = acc.float().cpu().numpy(); zf[idx] = np.nan
            else:
                q = o.clamp(min=0)
                out[idx] = q.mean(1).float().cpu().numpy()
                zf[idx] = (q < 1e-3).float().mean(1).cpu().numpy()
    npar = sum(p.numel() for p in m.parameters())
    del m; torch.cuda.empty_cache()
    return out, npar, (None if head == "gen" else float(np.nanmean(zf)))

NSEED = 2
print(f"\nФИНАЛЬНЫЙ протокол: обучение {SEL_TR}, предсказание на {SEL_EV}", flush=True)
assert SEL_EV + H > N_DAYS, "это не финальный якорь — у него есть цель, проверь протокол"
print(f"{'ствол':>8} {'сид':>4} {'парам':>9} {'нулей':>7} {'средняя log1p':>14} {'сек':>7}",
      flush=True)
for arm in ("cnn", "fusion"):
    acc = []
    for sd in range(NSEED):
        t0 = time.time()
        v, npar, zf = run(arm, "q", seed=sd)
        acc.append(v)
        print(f"{arm:>8} {sd:4d} {npar/1e6:8.2f}М {zf:7.3f} {v.mean():14.5f} "
              f"{time.time()-t0:7.0f}", flush=True)
    m = np.mean(acc, 0)
    np.save(OUT / f"p8_{arm}_q99_fin.npy", m)
    print(f"  {arm}: усреднено {NSEED} сидов, corr между сидами "
          f"{np.corrcoef(acc[0], acc[1])[0,1]:.5f} -> p8_{arm}_q99_fin.npy", flush=True)
print("\ndone")
