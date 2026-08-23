"""Скрытая марковская модель с hurdle-эмиссией: дискретный режим клиента.

Зачем именно она. Все 70+ векторов стека — гладкие функции признаков. HMM КОММИТИТСЯ к режиму, и
ошибка у неё устроена иначе: клиент далеко от границы состояний предсказывается точно, клиент у
границы — грубо. Такой формы ошибки у нас нет ни у одной компоненты, а стек платит именно за
инаковость ошибки (сегодня это подтвердилось восемь раз).

Главное возражение против HMM снимается тождеством Фруллани. Казалось бы, при марковской динамике
дни горизонта зависимы через путь состояний, и E[log1p(сумма)] требует перебора K^30 путей. Но
нам нужно не распределение суммы, а её преобразование Лапласа, а оно считается ПРЯМЫМ ХОДОМ:

    alpha_1(s)     = p_0(s) * L(t | s)
    alpha_{d+1}(s')= sum_s alpha_d(s) * A[s,s'] * L(t | s')
    L_S(t)         = sum_s alpha_30(s)

то есть O(30 * K^2) на клиента и на узел t, ТОЧНО, без симуляции. Дальше — та же квадратура
Фруллани, что уже работает. Модель новая, считыватель существующий.

Неоднородность клиентов — в ЛОГИТЕ, и это исправление первой версии.

Сначала я написал эмиссию как p = pi_s * (r_i / r_среднее) с обрезкой, а M-шаг — как простое
отношение взвешенных сумм. Это замкнутая форма для модели БЕЗ клиентского множителя и без
обрезки, то есть форма и обновление друг другу не соответствовали. Итог: правдоподобие пошло
вниз после первой итерации (-228,4 -> -87,0 -> -92,1 -> ... -> -92,7), а состояния схлопнулись
в почти одинаковые (pi от 0,125 до 0,148). Единственное, что уцелело, — структура переходов:
диагональ A [0,981, 0,555, 0,450, 0,421, 0,409], одно устойчивое состояние и четыре текучих.

Здесь p_{i,s} = sigmoid(a_s + b_i), где b_i = логит собственного темпа клиента (фиксирован), а
a_s обновляется шагом Ньютона по взвешенному бернуллиевому правдоподобию:

    dL/da_s  = sum gamma * (z - sigma),    d2L/da_s2 = -sum gamma * sigma * (1 - sigma)

Это обобщённый EM: он улучшает, а не максимизирует, и монотонность сохраняет. Обрезка не нужна
вовсе — сигмоида сама держит вероятность в (0,1). Правдоподобие обязано расти монотонно, и если
оно снова упадёт, значит ошибка в другом месте.

Ожидание по точности, записанное заранее: 1,75-1,78, то есть хуже наших 1,741. Это ровно та
полоса, где event_multi (1,74367) получил наибольший вес в четвёрке, а grn (1,77405) вошёл в
стек. Решает вклад в стенд, а не скор.
"""
import sys, os, time
sys.path.insert(0, "code")
import numpy as np
import torch
from common import panel, target, rmsle, NU, H, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = int(os.environ.get("K", 5))
ITERS = int(os.environ.get("ITERS", 12))
SEL_EV = int(os.environ.get("SEL_EV", 348))
CH = 20000
torch.manual_seed(0)

G = panel("gmv")[:, :SEL_EV + 1].astype(np.float32)
Z = (G > 0)
LG = np.where(Z, np.log1p(G), 0.0).astype(np.float32)
T = Z.shape[1]
r = Z.mean(1).astype(np.float32)                     # собственный темп клиента
rbar = float(r.mean())
cnt = np.maximum(Z.sum(1), 1)
cm = (LG.sum(1) / cnt).astype(np.float32)            # собственный средний чек
print(f"панель {Z.shape}, доля активных {Z.mean():.4f}, K={K}", flush=True)

b = torch.from_numpy(np.log(np.clip(r, 1e-4, 0.5) / (1 - np.clip(r, 1e-4, 0.5)))).to(DEV).double()
a = torch.linspace(-1.5, 1.5, K, device=DEV).double()      # сдвиг логита по состояниям
mu = torch.zeros(K, device=DEV).double()
sg = torch.full((K,), 1.0, device=DEV).double()
A = (torch.eye(K, device=DEV).double() * 6.0 + 1.0)
A = A / A.sum(1, keepdim=True)
p0 = torch.full((K,), 1.0 / K, device=DEV).double()

def logits(bc):
    return a.view(1, 1, -1) + bc.view(-1, 1, 1)                       # (B,T,K)

def emis(zc, lgc, bc, cmc):
    lo = logits(bc)
    lp1 = -torch.nn.functional.softplus(-lo)                          # log sigmoid
    lp0 = -torch.nn.functional.softplus(lo)                           # log(1-sigmoid)
    x = (lgc.unsqueeze(-1) - cmc.view(-1, 1, 1) - mu.view(1, 1, -1)) / sg.view(1, 1, -1)
    lpa = -0.5 * x * x - torch.log(sg.view(1, 1, -1)) - 0.9189385332
    return torch.where(zc.unsqueeze(-1), lp1 + lpa, lp0)

def fb(le):
    """прямой-обратный ход в логарифмах"""
    B, Tt, _ = le.shape
    lA = torch.log(A).unsqueeze(0)
    al = torch.empty(B, Tt, K, dtype=torch.float64, device=DEV)
    al[:, 0] = torch.log(p0).unsqueeze(0) + le[:, 0]
    for t_ in range(1, Tt):
        al[:, t_] = torch.logsumexp(al[:, t_ - 1].unsqueeze(-1) + lA, 1) + le[:, t_]
    be = torch.zeros(B, Tt, K, dtype=torch.float64, device=DEV)
    for t_ in range(Tt - 2, -1, -1):
        be[:, t_] = torch.logsumexp(lA + (le[:, t_ + 1] + be[:, t_ + 1]).unsqueeze(1), 2)
    ll = torch.logsumexp(al[:, -1], 1)
    gam = torch.softmax(al + be, dim=-1)
    xi = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    for t_ in range(Tt - 1):
        m_ = al[:, t_].unsqueeze(-1) + lA + (le[:, t_ + 1] + be[:, t_ + 1]).unsqueeze(1)
        xi += torch.exp(m_ - ll.view(-1, 1, 1)).sum(0)
    return gam, xi, ll.sum()

print(f"\n{'итерация':>9} {'логправдоподобие':>18} {'сек':>7}", flush=True)
for it in range(ITERS):
    t0 = time.time()
    nx = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    g1 = torch.zeros(K, dtype=torch.float64, device=DEV)     # первая производная по a_s
    g2 = torch.zeros(K, dtype=torch.float64, device=DEV)     # вторая (со знаком минус)
    s_w = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_x = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_x2 = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_p0 = torch.zeros(K, dtype=torch.float64, device=DEV)
    tot = 0.0
    for i in range(0, NU, CH):
        j = min(i + CH, NU)
        zc = torch.from_numpy(Z[i:j]).to(DEV)
        lgc = torch.from_numpy(LG[i:j]).to(DEV).double()
        bc = b[i:j]
        cmc = torch.from_numpy(cm[i:j]).to(DEV).double()
        gam, xi, ll = fb(emis(zc, lgc, bc, cmc))
        nx += xi; tot += float(ll)
        s_p0 += gam[:, 0].sum(0)
        sig = torch.sigmoid(logits(bc))
        g1 += (gam * (zc.unsqueeze(-1).double() - sig)).sum((0, 1))
        g2 += (gam * sig * (1.0 - sig)).sum((0, 1))
        gz = gam * zc.unsqueeze(-1)
        d = (lgc - cmc.view(-1, 1)).unsqueeze(-1)
        s_w += gz.sum((0, 1)); s_x += (gz * d).sum((0, 1)); s_x2 += (gz * d * d).sum((0, 1))
    A = (nx / nx.sum(1, keepdim=True)).clamp(1e-6, 1.0)
    A = A / A.sum(1, keepdim=True)
    p0 = (s_p0 / s_p0.sum()).clamp(1e-6, 1.0)
    a = a + g1 / g2.clamp(min=1e-9)          # шаг Ньютона, целевая по a_s вогнута
    mu = s_x / s_w.clamp(min=1e-9)
    sg = ((s_x2 / s_w.clamp(min=1e-9)) - mu * mu).clamp(min=0.01).sqrt()
    print(f"{it:>9} {tot/NU:18.5f} {time.time()-t0:7.0f}", flush=True)

pv = torch.sigmoid(a + float(np.log(0.05 / 0.95))).cpu().numpy()
pv = torch.sigmoid(a + float(np.log(0.05 / 0.95))).cpu().numpy()
print(f"\nсостояния: a={np.round(a.cpu().numpy(),3)}")
print(f"           p при среднем клиенте={np.round(pv,4)}")
print(f"           mu={np.round(mu.cpu().numpy(),3)}  sigma={np.round(sg.cpu().numpy(),3)}")
dg = torch.diag(A).cpu().numpy()
print(f"диагональ A: {np.round(dg,4)}   время автокорреляции режима "
      f"{1.0/(1.0-float(dg.mean())):.2f} дня")

# ---- считыватель: тождество Фруллани поверх прямого хода по горизонту ----
# L_S(t) = sum_s alpha_H(s), где alpha идёт по дням горизонта с матрицей переходов A,
# а L(t | s, i) = (1-p) + p * E[exp(-t*Y)] при Y = expm1(cm_i + mu_s + sg_s*Z).
# Перебора K^30 путей не нужно: прямой ход даёт то же самое точно, за O(H*K^2) на узел t.
NQT = 128
_gx, _gw = np.polynomial.legendre.leggauss(NQT)
_u = 0.5 * (_gx + 1) * (14.0 - (-32.0)) + (-32.0)
T_NODES = torch.tensor(np.exp(_u), dtype=torch.float64, device=DEV)
T_W = torch.tensor(_gw * 0.5 * (14.0 - (-32.0)) * np.exp(_u), dtype=torch.float64, device=DEV)
zz, ww = np.polynomial.hermite_e.hermegauss(24)
zz = torch.tensor(zz, dtype=torch.float64, device=DEV)
ww = torch.tensor(ww / ww.sum(), dtype=torch.float64, device=DEV)

print("\nсчитываю горизонт прямым ходом ...", flush=True)
out = np.empty(NU, np.float32)
CH2 = 4000
t0 = time.time()
with torch.no_grad():
    for i0 in range(0, NU, CH2):
        j0 = min(i0 + CH2, NU)
        zc = torch.from_numpy(Z[i0:j0]).to(DEV)
        lgc = torch.from_numpy(LG[i0:j0]).to(DEV).double()
        bc = b[i0:j0]
        cmc = torch.from_numpy(cm[i0:j0]).to(DEV).double()
        le = emis(zc, lgc, bc, cmc)
        lA = torch.log(A).unsqueeze(0)
        al = torch.log(p0).unsqueeze(0) + le[:, 0]
        for t_ in range(1, le.shape[1]):
            al = torch.logsumexp(al.unsqueeze(-1) + lA, 1) + le[:, t_]
        post = torch.softmax(al, dim=-1)                                  # (B,K) на дне 348
        p_is = torch.sigmoid(a.view(1, -1) + bc.view(-1, 1))              # (B,K)
        yv = torch.expm1(cmc.view(-1, 1, 1) + mu.view(1, -1, 1)
                         + sg.view(1, -1, 1) * zz.view(1, 1, -1)).clamp(min=0.0)   # (B,K,GH)
        lam = (torch.exp(-T_NODES.view(1, 1, 1, -1) * yv.unsqueeze(-1))
               * ww.view(1, 1, -1, 1)).sum(2)                             # (B,K,NT)
        Lts = (1.0 - p_is).unsqueeze(-1) + p_is.unsqueeze(-1) * lam       # (B,K,NT)
        al2 = post.unsqueeze(-1).expand(-1, -1, T_NODES.shape[0]).clone()   # (B,K,NT)
        for _ in range(H):
            al2 = torch.einsum("bkt,ks->bst", al2, A) * Lts
        LS = al2.sum(1)                                                   # (B,NT)
        val = ((1.0 - LS) * torch.exp(-T_NODES) / T_NODES * T_W).sum(-1)
        out[i0:j0] = val.float().cpu().numpy()
print(f"  готово за {time.time()-t0:.0f}s", flush=True)
np.save(OUT / f"x7_hmm_K{K}.npy", out)

yt = target(SEL_EV); lyt = np.log1p(yt)
sh = out - out.mean() + lyt.mean()
print(f"\nHMM K={K}: RMSLE {rmsle(yt, np.expm1(np.clip(sh,0,13))):.5f}  -> x7_hmm_K{K}.npy")
for nm in ("h8_gauss", "q2c_daily_sel", "v3_gru96_visit"):
    f = OUT / f"{nm}.npy"
    if f.exists():
        r = np.load(f)
        if r.shape == (NU,):
            print(f"  corr с {nm}: {np.corrcoef(out, r)[0,1]:.5f}")
print("\nвклад считается отдельно: w11 с WHO=x7_hmm_K5")
print("\ndone")
