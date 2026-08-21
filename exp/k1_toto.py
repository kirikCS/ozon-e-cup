"""Toto 2.0 zero-shot: предобученный ствол против нашей дневной генеративной модели.

Ключевое обстоятельство, снявшее главное возражение: Toto декодерный и авторегрессионный, а
`generate_samples` порождает СОВМЕСТНЫЕ пути, а не независимые маргиналы по дням. Значит
E[log1p(Σ_d y_d)] берётся честно — усреднением log1p от суммы каждого пути, — и разрыв Йенсена
(измеренный у нас как 1.0906) не возникает.

Это делает Toto прямым конкурентом нашей дневной модели: та же схема (симуляция путей, оптимум
RMSLE через среднее логарифма суммы), но ствол предобучен на телеметрии Datadog вместо обучения
с нуля на наших якорях.

Два известных риска, которые тест и должен взвесить:
  ДОМЕН. Обучение на метриках наблюдаемости — плотные положительные ряды. У нас 95% нулей на
  дневном уровне; классика прерывистого спроса на этих данных давала не лучше 1.8405.
  МАСШТАБ. 250k клиентов × 30 шагов авторегрессии × N путей. Скорость меряется явно, до выводов.

Три варианта входа:
  uni      только дневной gmv
  multi3   gmv, визиты, корзины — Toto умеет внимание по оси переменных
  log      log1p(gmv), на случай если arcsinh-скейлер плохо ест нулевую массу

Сравнение честное: та же подвыборка клиентов, тот же якорь 348, тот же уровень выровнен.
"""
import sys, time, os
import numpy as np
import torch

sys.path.insert(0, "code")
from common import panel, target, rmsle, NU, OUT
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster
from toto.data.util.dataset import MaskedTimeseries

DEV = "cuda"
A, H = 348, 30
CTX = 364
NSUB = int(os.environ.get("NSUB", 2000))
NSAMP = int(os.environ.get("NSAMP", 128))
BS = int(os.environ.get("BS", 16))
MODEL = os.environ.get("TOTO_MODEL", "Datadog/Toto-2.0-313m")

G = panel("gmv").astype(np.float32)
V = panel("visit").astype(np.float32)
C = panel("cart").astype(np.float32)
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(NU, NSUB, replace=False))
y = target(A)[sub]
ly = np.log1p(y)
print(f"модель {MODEL}, подвыборка {NSUB}, путей {NSAMP}, батч {BS}", flush=True)
print(f"доля покупающих в подвыборке: {(y>0).mean():.4f}", flush=True)

print("загружаю ...", flush=True)
t0 = time.time()
model = Toto.from_pretrained(MODEL).to(DEV)
model.eval()
fc = TotoForecaster(model.model)
print(f"  {time.time()-t0:.0f}s, параметров {sum(p.numel() for p in model.parameters())/1e6:.0f}М",
      flush=True)

BASE_TS = 1735689600          # 2025-01-01 в unix-секундах
DAY = 86400

def run_variant(tag, chans, logify):
    nv = len(chans)
    out = np.empty(NSUB, np.float32)
    t0 = time.time()
    for i in range(0, NSUB, BS):
        u = sub[i:i + BS]
        b = len(u)
        arr = np.stack([M[u, A - CTX + 1:A + 1] for M in chans], 1).astype(np.float32)
        if logify:
            arr = np.log1p(np.maximum(arr, 0))
        x = torch.from_numpy(arr).to(DEV)
        ts = torch.arange(A - CTX + 1, A + 1, device=DEV) * DAY + BASE_TS
        ts = ts[None, None, :].expand(b, nv, CTX).to(torch.int64)
        mts = MaskedTimeseries(
            series=x,
            padding_mask=torch.ones_like(x, dtype=torch.bool),
            id_mask=torch.zeros(b, nv, CTX, dtype=torch.int64, device=DEV),
            timestamp_seconds=ts,
            time_interval_seconds=torch.full((b, nv), DAY, dtype=torch.int64, device=DEV))
        with torch.no_grad():
            f = fc.forecast(mts, prediction_length=H, num_samples=NSAMP,
                            samples_per_batch=min(NSAMP, 32), use_kv_cache=True)
        s = f.samples                       # (batch, variate, time, samples)
        s = s[:, 0]                         # целевая переменная — первая
        if logify:
            s = torch.expm1(s)
        tot = s.clamp(min=0).sum(1)         # (batch, samples) — сумма по 30 дням каждого пути
        out[i:i + b] = torch.log1p(tot).mean(-1).float().cpu().numpy()
        if i == 0:
            el = time.time() - t0
            print(f"  [{tag}] первый батч {el:.1f}s -> на 250k примерно "
                  f"{el*250000/BS/3600:.1f} ч", flush=True)
    sh = out - out.mean() + ly.mean()
    r = rmsle(y, np.expm1(np.clip(sh, 0, 13)))
    print(f"  [{tag}] RMSLE {r:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return out, r

REF = OUT / "h8_gauss.npy"
if REF.exists():
    ref = np.load(REF)[sub]
    ref = ref - ref.mean() + ly.mean()
    print(f"\nнаша дневная модель на этой подвыборке: "
          f"{rmsle(y, np.expm1(np.clip(ref,0,13))):.5f}", flush=True)

print("\n=== варианты входа ===", flush=True)
RES = {}
for tag, chans, lo in (("uni", [G], False), ("log", [G], True), ("multi3", [G, V, C], False)):
    try:
        v, r = run_variant(tag, chans, lo)
        RES[tag] = v
        np.save(OUT / f"k1_toto_{tag}.npy", v)
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"  [{tag}] ОШИБКА {type(e).__name__}: {str(e)[:120]}", flush=True)
if REF.exists() and RES:
    print("\ncorr с нашей дневной моделью:")
    for tag, v in RES.items():
        print(f"  {tag:8s} {np.corrcoef(v, np.load(REF)[sub])[0,1]:.5f}")
print("\ndone")
