"""Toto 2.0 zero-shot с копулой: маргиналы от предобученной модели, зависимость наша.

Поправка к предыдущей версии теста. У Toto 2.0 НЕТ generate_samples — этот метод был в 1.0,
чью сигнатуру я по ошибке и посмотрел. Публичный API 2.0 отдаёт КВАНТИЛИ (9, batch, n_var,
horizon), обученные pinball loss, а decode_block_size и «masked contiguous patches for parallel
decoding» означают параллельное декодирование, а не авторегрессионное.

Значит квантили МАРГИНАЛЬНЫЕ по дням, и распределение суммы из них не восстанавливается. Это то
самое возражение, которое мы уже проверяли на своих данных: независимая выборка дней завышает
P(Y>0) с 0.555 до 0.684, а наивная сборка даёт 2.48 против 1.755 у симуляции с зависимостью.

Решение — гауссова копула. Маргиналы берём от Toto, зависимость добавляем сами:

    u_d = Φ( ρ·z + sqrt(1-ρ²)·ε_d ),   z общая на путь, ε_d независимы
    y_d = F_d^{-1}(u_d)                 обратная функция по девяти квантилям

При ρ = 0 это независимость, при ρ → 1 все дни двигаются вместе. Параметр один, подбирается по
совпадению P(Y>0) с эмпирической — тот же приём, что дал нашей дневной модели −0.006.

Три руки: сырой gmv, log1p(gmv) на случай если arcsinh плохо ест нулевую массу, и трёхмерный
вход через внимание по оси переменных.
"""
import sys, time, os
import numpy as np
import torch

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT
from toto2 import Toto2Model

DEV = "cuda"
A, H = 348, 30
# patch_size у Toto 2.0 равен 32, контекст берём кратным ему и НЕ длиннее доступной истории:
# при якоре 348 доступно 349 дней, поэтому 320 = 10 патчей. Прошлая версия просила 512,
# старт окна уходил в -163, и numpy молча возвращал 103 дня вместо 512.
CTX = int(os.environ.get("CTX", 320))
NSUB = int(os.environ.get("NSUB", 2000))
NPATH = int(os.environ.get("NPATH", 256))
BS = int(os.environ.get("BS", 64))
MODEL = os.environ.get("TOTO_MODEL", "Datadog/Toto-2.0-313m")
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(NU, NSUB, replace=False))
y = target(A)[sub]; ly = np.log1p(y)
emp_pos = float((y > 0).mean())
print(f"{MODEL}, подвыборка {NSUB}, путей {NPATH}, батч {BS}, контекст {CTX}", flush=True)
print(f"эмпирическая P(Y>0) = {emp_pos:.4f}", flush=True)

t0 = time.time()
model = Toto2Model.from_pretrained(MODEL).to(DEV).eval()
print(f"загружено за {time.time()-t0:.0f}s, "
      f"{sum(p.numel() for p in model.parameters())/1e6:.0f}М параметров", flush=True)

def get_quantiles(chans, logify):
    """квантили на 30 дней вперёд для всей подвыборки: (NSUB, 9, H)"""
    nv = len(chans)
    out = np.empty((NSUB, 9, H), np.float32)
    t0 = time.time()
    for i in range(0, NSUB, BS):
        u = sub[i:i + BS]; b = len(u)
        lo_i = A - CTX + 1
        if lo_i >= 0:
            arr = np.stack([M[u, lo_i:A + 1] for M in chans], 1).astype(np.float32)
        else:
            got = np.stack([M[u, 0:A + 1] for M in chans], 1).astype(np.float32)
            arr = np.concatenate([np.zeros((b, nv, -lo_i), np.float32), got], 2)
        assert arr.shape == (b, nv, CTX), f"окно {arr.shape}, ожидалось {(b, nv, CTX)}"
        if logify:
            arr = np.log1p(np.maximum(arr, 0))
        tgt = torch.from_numpy(arr).to(DEV)
        with torch.no_grad():
            q = model.forecast(
                {"target": tgt,
                 "target_mask": torch.ones_like(tgt, dtype=torch.bool),
                 "series_ids": torch.zeros(b, nv, dtype=torch.long, device=DEV)},
                horizon=H, decode_block_size=768, has_missing_values=False)
        qq = q[:, :, 0] if q.ndim == 4 else q          # (9, b, H)
        out[i:i + b] = qq.permute(1, 0, 2).float().cpu().numpy()
        if i == 0:
            el = time.time() - t0
            print(f"    первый батч {el:.1f}s -> на 250k примерно {el*250000/b/3600:.1f} ч",
                  flush=True)
    return out, time.time() - t0

def simulate(Q, rho, logify, seed=0):
    """гауссова копула поверх маргиналов: общий латент на путь + обратная функция по квантилям"""
    r = np.random.default_rng(seed)
    z = r.standard_normal((NPATH, NSUB, 1))
    e = r.standard_normal((NPATH, NSUB, H))
    g = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * e
    from scipy.stats import norm
    u = norm.cdf(g)                                     # (NPATH, NSUB, H)
    # обратная функция: линейная интерполяция по девяти квантилям, хвосты обрезаны
    Qs = np.sort(Q, axis=1)
    idx = np.clip(np.searchsorted(QL, u.ravel()), 1, 8)
    lo, hi = QL[idx - 1], QL[idx]
    w = ((u.ravel() - lo) / (hi - lo)).astype(np.float32)
    flat = np.broadcast_to(Qs[None], (NPATH, NSUB, 9, H))
    ii = np.arange(NPATH * NSUB * H)
    p_ = flat.transpose(0, 1, 3, 2).reshape(-1, 9)
    val = p_[ii, idx - 1] * (1 - w) + p_[ii, idx] * w
    val = val.reshape(NPATH, NSUB, H)
    if logify:
        val = np.expm1(val)
    tot = np.maximum(val, 0).sum(-1)                    # (NPATH, NSUB)
    return np.log1p(tot).mean(0), float((tot > 0).mean())

REF = OUT / "h8_gauss.npy"
if REF.exists():
    ref = np.load(REF)[sub]; ref = ref - ref.mean() + ly.mean()
    print(f"\nнаша дневная модель на этой подвыборке: "
          f"{rmsle(y, np.expm1(np.clip(ref,0,13))):.5f}", flush=True)

print("\n=== варианты входа ===", flush=True)
for tag, chans, lo_ in (("uni", [G], False), ("log", [G], True), ("multi3", [G, V, C], False)):
    try:
        Q, el = get_quantiles(chans, lo_)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"  [{tag}] ОШИБКА {type(e).__name__}: {str(e)[:130]}", flush=True)
        continue
    print(f"  [{tag}] квантили за {el:.0f}s; подбираю ρ по совпадению P(Y>0)", flush=True)
    best = None
    for rho in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95):
        v, pos = simulate(Q, rho, lo_)
        sh = v - v.mean() + ly.mean()
        r = rmsle(y, np.expm1(np.clip(sh, 0, 13)))
        print(f"      ρ={rho:.2f}  P(Y>0)={pos:.4f}  RMSLE {r:.5f}", flush=True)
        if best is None or abs(pos - emp_pos) < best[0]:
            best = (abs(pos - emp_pos), rho, v, r)
    _, rho, v, r = best
    print(f"  [{tag}] выбрано ρ={rho:.2f}, RMSLE {r:.5f}", flush=True)
    np.save(OUT / f"k2_toto_{tag}.npy", v)
    if REF.exists():
        print(f"  [{tag}] corr с нашей дневной {np.corrcoef(v, np.load(REF)[sub])[0,1]:.5f}",
              flush=True)
print("\ndone")
