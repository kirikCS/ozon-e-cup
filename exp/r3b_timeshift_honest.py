"""Time shifts, done honestly. The first version leaked and its numbers are void.

The defect: it moved the WINDOW while leaving the label attached to the original anchor. For a
positive shift the window then reaches into the target window and reads the answer — which is
why "+14" scored 1.5466 against a base of 1.7713, a number that cannot exist. The augmentation
arms carried the same leak into training, which is why the week-preserving shifts came out
WORSE: the model learned to lean on days it would not have at evaluation.

Done properly, a shift is a NEW ANCHOR: window and label move together, each anchor keeping its
own strictly-historical window and its own target. And only backwards — anchor 318 plus 30 days
is exactly the evaluation anchor 348, so any forward shift eats into the protocol's 30-day gap
and leaks the evaluation label.

What this then measures is real and still interesting: the existing anchors step by 28 days
(4 weeks, weekday-aligned by design). Shifts of -7 and -14 keep that alignment while overlapping
the target windows; shifts of -3 and -10 break it. If only the aligned ones help, the weekly
structure is what carries; if both help equally, it is regularisation by correlated labels; if
neither helps, the anchor plateau the teammate already measured at 10 anchors is real and the
spacing is not the binding constraint.

The diagnostic is rebuilt as an honest invariance test: two windows of EQUAL distance from the
anchor, one weekday-aligned and one not, so the comparison is not confounded by recency.
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

def win(users, end, length=L):
    s = end - length + 1
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

def train(anchors, seed=0):
    torch.manual_seed(seed)
    m = CNN(CH).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * len(anchors) * (NU // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 10))
    Y = {a: target(a).astype(np.float32) for a in anchors}
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        m.train()
        for a in anchors:
            perm = rng.permutation(NU)
            for i in range(0, NU, BS):
                idx = np.sort(perm[i:i + BS])
                if len(idx) < 64:
                    continue
                x = torch.from_numpy(win(idx, a)).to(DEV)
                y = torch.from_numpy(Y[a][idx]).to(DEV)
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
def predict(m, anchor, length=L):
    out = []
    for i in range(0, NU, 8192):
        idx = np.arange(i, min(i + 8192, NU))
        o = m(torch.from_numpy(win(idx, anchor, length)).to(DEV))
        out.append((torch.sigmoid(o[:, 0]) * o[:, 1].clamp(min=0)).cpu().numpy())
    return np.concatenate(out)

y = target(SEL_EV)
print("=== 1. инвариантность: равное расстояние, разное выравнивание по дню недели ===")
print("   (все окна строго до якоря — сравниваются пары одинаковой давности)")
m0 = train(SEL_TR, 0)
p0 = predict(m0, SEL_EV)
print(f"  опорное окно, конец на якоре: RMSLE {rmsle(y, np.expm1(np.clip(p0,0,13))):.5f}\n")
print(f"  {'длина окна':>12} {'выравнивание':>14} {'corr с опорным':>16} {'RMSLE':>9}")
for length in (175, 178, 182, 189):
    pk = predict(m0, SEL_EV, length)
    al = "кратна неделе" if length % 7 == 0 else "не кратна"
    print(f"  {length:12d} {al:>14} {np.corrcoef(p0,pk)[0,1]:16.5f} "
          f"{rmsle(y, np.expm1(np.clip(pk,0,13))):9.5f}")
del m0
torch.cuda.empty_cache()

print("\n=== 2. аугментация новыми якорями (окно и метка едут вместе, только назад) ===")
print(f"  {'набор':40s} {'якорей':>7} {'RMSLE':>9} {'к базе':>9}")
base_r = None
for tag, shifts in (("базовые 5 якорей", [0]),
                    ("+ сдвиг -7 (день недели сохранён)", [0, -7]),
                    ("+ сдвиги -7, -14 (сохранён)", [0, -7, -14]),
                    ("+ сдвиг -3 (день недели сломан)", [0, -3]),
                    ("+ сдвиги -3, -10 (сломан)", [0, -3, -10])):
    anchors = sorted({a + s for a in SEL_TR for s in shifts if a + s >= 60})
    assert max(anchors) + 30 <= SEL_EV, "нарушен тридцатидневный зазор протокола"
    r = rmsle(y, np.expm1(np.clip(predict(train(anchors, 0), SEL_EV), 0, 13)))
    if base_r is None:
        base_r = r
    print(f"  {tag:40s} {len(anchors):7d} {r:9.5f} {r-base_r:+9.5f}")
    torch.cuda.empty_cache()
print("\nчитать так: помогают только кратные неделе — работает недельная структура;")
print("помогают одинаково — это регуляризация коррелированными метками; не помогают — плато по якорям")
print("done")
