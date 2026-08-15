"""Move clients along the timeline: does WHEN a purchase happened matter beyond how long ago?

Two things, and they answer different questions.

DIAGNOSTIC — is the model calendar-invariant? Take the evaluation users, shift their whole
history by k days, ask the model again, and measure how far the prediction moves. A model that
only reads recency and volume barely moves; one that has latched onto absolute calendar
position moves a lot. The contrast between shifts of 7 days (weekday alignment preserved) and
of 1-3 days (alignment broken) separates "day-of-week structure" from "absolute position".

AUGMENTATION — if the process is shift-invariant up to weekday, then shifting is free training
data. The existing anchors step by 28 days, i.e. exactly 4 weeks, which already preserves
weekday and is why the geometry was chosen. Shifts of +-7 and +-14 days multiply the anchor
set without breaking that alignment, and without touching the 30-day gap between the last
training anchor and the evaluation window. Shifts of +-3 days break alignment and act as the
control: if the augmentation gain comes from weekday-preserving shifts only, the effect is real
structure; if both help equally, it is plain regularisation by noise.

The augmentation is run on the pure-sequence CNN, which is the model that actually reads the
daily series — a tree consuming window aggregates is nearly shift-invariant by construction.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, N_DAYS, ds

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, EPOCHS, BS, LR = 182, 5, 2048, 1.5e-3
SEL_TR, SEL_EV = [318, 290, 262, 234, 206], 348

def log16(n):
    return np.log1p(panel(n).astype(np.float32)).astype(np.float16)

SEQ = np.stack([log16("gmv"), log16("gmv_search"), log16("ord"), log16("cart"), log16("srch"),
                panel("visit").astype(np.float16), panel("engaged").astype(np.float16)], 1)
CH = SEQ.shape[1]

def win(users, end):
    s = end - L + 1
    if s >= 0:
        return SEQ[users, :, s:end + 1].astype(np.float32)
    got = SEQ[users, :, 0:end + 1].astype(np.float32)
    return np.concatenate([np.zeros((len(users), CH, -s), np.float32), got], 2)

class CNN(nn.Module):
    def __init__(self, c, h=96, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(c, h, 3, padding=1)
        self.bl = nn.ModuleList([nn.Sequential(
            nn.Conv1d(h, h, 3, padding=d, dilation=d), nn.BatchNorm1d(h), nn.SiLU()) for d in dil])
        self.head = nn.Sequential(nn.Linear(2 * h, 128), nn.BatchNorm1d(128), nn.SiLU(),
                                  nn.Linear(128, 2))
    def forward(self, x):
        z = self.inp(x)
        for b in self.bl:
            z = z + b(z)
        return self.head(torch.cat([z.mean(-1), z.max(-1).values], -1))

def train(anchor_list, seed=0):
    """anchor_list: [(anchor, target_anchor)] — the window may be shifted while the label
    stays attached to its own anchor, which is what makes this an augmentation"""
    torch.manual_seed(seed)
    m = CNN(CH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(anchor_list) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {t: target(t).astype(np.float32) for _, t in anchor_list}
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        m.train()
        for wa, ta in anchor_list:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, wa)).to(DEV)
                y = torch.from_numpy(Y[ta][idx]).to(DEV)
                o = m(x)
                pos = (y > 0).float()
                loss = (nn.functional.binary_cross_entropy_with_logits(o[:, 0], pos,
                                                                       reduction="none")
                        + pos * (o[:, 1] - torch.log1p(y)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
                if sch.last_epoch < steps - 1:
                    sch.step()
    return m.eval()

@torch.no_grad()
def predict(m, anchor):
    out = []
    for i in range(0, NU, 8192):
        idx = np.arange(i, min(i + 8192, NU))
        o = m(torch.from_numpy(win(idx, anchor)).to(DEV))
        out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    return np.concatenate(out)

y = target(SEL_EV)
print(f"=== 1. диагностика: насколько модель чувствительна к сдвигу окна ===")
base_pairs = [(a, a) for a in SEL_TR]
m0 = train(base_pairs, 0)
p0 = predict(m0, SEL_EV)
print(f"  без сдвига: RMSLE {rmsle(y, np.expm1(np.clip(p0,0,13))):.5f}\n")
print(f"  {'сдвиг окна':>12} {'corr с исходным':>16} {'сдвиг среднего':>15} {'RMSLE':>9}")
for k in (-14, -7, -3, -1, 1, 3, 7, 14):
    pk = predict(m0, SEL_EV + k) if SEL_EV + k < N_DAYS else None
    if pk is None:
        continue
    tag = "неделя" if k % 7 == 0 else "ломает дни"
    print(f"  {k:+12d} {np.corrcoef(p0,pk)[0,1]:16.5f} {pk.mean()-p0.mean():+15.4f} "
          f"{rmsle(y, np.expm1(np.clip(pk,0,13))):9.5f}   {tag}")
del m0
torch.cuda.empty_cache()

print(f"\n=== 2. аугментация сдвигами ===")
print(f"  {'набор':38s} {'якорей':>7} {'RMSLE':>9} {'к базе':>9}")
base_r = None
for tag, shifts in (("без аугментации", [0]),
                    ("+-7 дней (день недели сохранён)", [0, -7, 7]),
                    ("+-7, +-14 (день недели сохранён)", [0, -7, 7, -14, 14]),
                    ("+-3 дня (день недели сломан)", [0, -3, 3]),
                    ("+-3, +-10 (день недели сломан)", [0, -3, 3, -10, 10])):
    pairs = [(a + s, a) for a in SEL_TR for s in shifts if 0 <= a + s < N_DAYS]
    m = train(pairs, 0)
    r = rmsle(y, np.expm1(np.clip(predict(m, SEL_EV), 0, 13)))
    if base_r is None:
        base_r = r
    print(f"  {tag:38s} {len(pairs):7d} {r:9.5f} {r-base_r:+9.5f}")
    del m
    torch.cuda.empty_cache()
print("\nчитать так: если +-7 помогает, а +-3 нет — работает структура недели, а не шум")
print("done")
