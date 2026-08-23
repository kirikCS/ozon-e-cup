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

Неоднородность клиентов. Глобальная HMM дала бы всем одинаковую динамику. Поэтому эмиссия
масштабируется собственным темпом клиента: вероятность покупки в состоянии s у клиента i равна
clip(pi_s * r_i / r_срednee), а величина чека сдвигается на собственное среднее клиента. EM при
этом остаётся простым: подбираются pi_s, mu_s, sigma_s и матрица переходов A.

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

pi = torch.linspace(0.2, 2.5, K, device=DEV).double()        # множитель темпа по состояниям
mu = torch.zeros(K, device=DEV).double()                     # сдвиг чека по состояниям
sg = torch.full((K,), 1.0, device=DEV).double()
A = (torch.eye(K, device=DEV).double() * 6.0 + 1.0)
A = A / A.sum(1, keepdim=True)
p0 = torch.full((K,), 1.0 / K, device=DEV).double()

def emis(zc, lgc, rc, cmc):
    """логарифм вероятности наблюдения в каждом состоянии: (B, T, K)"""
    pr = (pi.view(1, 1, -1) * (rc / rbar).view(-1, 1, 1)).clamp(1e-4, 0.98)
    lp0 = torch.log1p(-pr)
    x = (lgc.unsqueeze(-1) - cmc.view(-1, 1, 1) - mu.view(1, 1, -1)) / sg.view(1, 1, -1)
    lp1 = torch.log(pr) - 0.5 * x * x - torch.log(sg.view(1, 1, -1)) - 0.9189385332
    return torch.where(zc.unsqueeze(-1), lp1, lp0)

def fb(le):
    """прямой-обратный ход в логарифмах: возвращает гамму, кси-сумму и правдоподобие"""
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
    return gam, xi, ll.sum(), al[:, -1]

print(f"\n{'итерация':>9} {'логправдоподобие':>18} {'сек':>7}", flush=True)
for it in range(ITERS):
    t0 = time.time()
    nx = torch.zeros(K, K, dtype=torch.float64, device=DEV)
    s_g = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_gz = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_w = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_x = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_x2 = torch.zeros(K, dtype=torch.float64, device=DEV)
    s_p0 = torch.zeros(K, dtype=torch.float64, device=DEV)
    tot = 0.0
    for i in range(0, NU, CH):
        j = min(i + CH, NU)
        zc = torch.from_numpy(Z[i:j]).to(DEV)
        lgc = torch.from_numpy(LG[i:j]).to(DEV).double()
        rc = torch.from_numpy(r[i:j]).to(DEV).double()
        cmc = torch.from_numpy(cm[i:j]).to(DEV).double()
        gam, xi, ll, _ = fb(emis(zc, lgc, rc, cmc))
        nx += xi; tot += float(ll)
        s_p0 += gam[:, 0].sum(0)
        w = gam * (rc / rbar).view(-1, 1, 1)
        s_g += w.sum((0, 1))
        s_gz += (w * zc.unsqueeze(-1)).sum((0, 1))
        gz = gam * zc.unsqueeze(-1)
        d = (lgc - cmc.view(-1, 1)).unsqueeze(-1)
        s_w += gz.sum((0, 1)); s_x += (gz * d).sum((0, 1)); s_x2 += (gz * d * d).sum((0, 1))
    A = (nx / nx.sum(1, keepdim=True)).clamp(1e-6, 1.0)
    A = A / A.sum(1, keepdim=True)
    p0 = (s_p0 / s_p0.sum()).clamp(1e-6, 1.0)
    pi = (s_gz / s_g.clamp(min=1e-9)).clamp(1e-4, 0.98)
    mu = s_x / s_w.clamp(min=1e-9)
    sg = ((s_x2 / s_w.clamp(min=1e-9)) - mu * mu).clamp(min=0.01).sqrt()
    print(f"{it:>9} {tot/NU:18.5f} {time.time()-t0:7.0f}", flush=True)

print(f"\nсостояния: pi={np.round(pi.cpu().numpy(),4)}")
print(f"           mu={np.round(mu.cpu().numpy(),3)}  sigma={np.round(sg.cpu().numpy(),3)}")
print(f"диагональ A: {np.round(torch.diag(A).cpu().numpy(),4)}")
np.save(OUT / f"x5_hmm_states_K{K}.npy", np.stack([pi.cpu().numpy(), mu.cpu().numpy(),
                                                   sg.cpu().numpy()]))
print("\ndone")
