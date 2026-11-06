"""Чистый BTYD-предиктор: BG/NBD × Gamma-Gamma = E[GMV_30d].

Полностью параметрическая модель, без нейронок. Ожидаем низкую corr с evgru3
(разная функциональная форма, разный тип входных данных).

BT[a][:,0] = E[transactions next 30d]  (BG/NBD)
BT[a][:,2] = E[order value]            (Gamma-Gamma)
=> BTYD_pred = col0 * col2             = E[GMV_30d]

Дополнительно: P(alive) [col1] как мультипликатор → несколько вариантов.

Якорь 348 → local eval. Якорь 408 → LB submission.

Usage: python -u code/fx_btyd.py
"""
import sys, pickle, time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import NU, target, rmsle, OUT

EL = 2.3284

print("loading BTYD cache...", flush=True)
BT = pickle.load(open(OUT / "btyd_feats.pkl", "rb"))
uids = np.load(OUT.parent / "cache" / "uids.npy")

# Ближайший к ev3 набор для corr-контроля
l2s = [42, 101, 777, 7, 1234, 2024, 31337, 555, 9090]
ens_2s = np.mean([np.load(OUT / f"fx_se_2s_s{s}.npy") for s in l2s], 0)
ev3    = np.mean([np.load(OUT / f"fx_evgru3_s{s}.npy") for s in [0, 1]], 0)

EV = 348
Y_EV = target(EV)

bt_ev = BT[EV].astype(np.float64)     # (NU, 5)
E_n   = bt_ev[:, 0]   # E[#transactions in next 30d]
P_a   = bt_ev[:, 1]   # P(alive)
E_v   = bt_ev[:, 2]   # E[order value]
rate  = bt_ev[:, 3]   # historical purchase rate

# Несколько вариантов предсказания
variants = {
    "btyd_basic":  E_n * E_v,                    # E[N] × E[V]
    "btyd_palive": P_a * E_n * E_v,              # P(alive) × E[N] × E[V]
    "btyd_log":    np.expm1(bt_ev[:, 4]),         # EB log-blocks (col4)
}

print(f"\n=== Local eval at EV={EV} ===", flush=True)
best_name, best_sc = None, 9.9
for name, pred_gmv in variants.items():
    pred_gmv = np.clip(np.nan_to_num(pred_gmv), 0, None)
    pred_lp = np.log1p(pred_gmv)

    # Level correction
    pred_lp_c = pred_lp - pred_lp.mean() + EL

    sc = rmsle(Y_EV, np.expm1(np.clip(pred_lp_c, 0, 13)))
    c1 = np.corrcoef(pred_lp_c, ens_2s)[0, 1]
    c2 = np.corrcoef(pred_lp_c, ev3)[0, 1]
    survived = (sc <= 1.820) and (c2 <= 0.985)
    verdict  = "ВЫЖИЛ" if survived else "закрыт"
    print(f"  {name}: score={sc:.5f}  corr(2s)={c1:.5f}  corr(ev3)={c2:.5f}  → {verdict}", flush=True)
    if sc < best_sc:
        best_sc, best_name = sc, name

print(f"\nЛучший: {best_name} (score={best_sc:.5f})", flush=True)

# LB submission at anchor 408
LB_ANCHOR = 408
if LB_ANCHOR in BT:
    bt_lb = BT[LB_ANCHOR].astype(np.float64)
    E_n_lb = bt_lb[:, 0]; P_a_lb = bt_lb[:, 1]; E_v_lb = bt_lb[:, 2]

    # Для каждого варианта пишем LB submission
    preds_lb = {
        "btyd_basic":  E_n_lb * E_v_lb,
        "btyd_palive": P_a_lb * E_n_lb * E_v_lb,
        "btyd_log":    np.expm1(bt_lb[:, 4]),
    }
    for name, pred_lb in preds_lb.items():
        pred_lb = np.clip(np.nan_to_num(pred_lb), 0, None)
        pred_lp = np.log1p(pred_lb)
        pred_lp = pred_lp - pred_lp.mean() + EL
        pv = np.clip(np.expm1(pred_lp), 0, None)

        out_path = Path(f"submissions/btyd_{name}_lb.csv")
        with open(out_path, "w") as f:
            f.write("user_id,predict\n")
            for u, v in zip(uids, pv):
                f.write(f"{u},{v:.6f}\n")
        print(f"  wrote {out_path}  (mean={pred_lp.mean():.4f})", flush=True)

    # Проверяем корреляцию между BTYD_lb и ev3 LB
    ev3_path = "submissions/evgru3_final_lb1656201.csv"
    if Path(ev3_path).exists():
        d = pl.read_csv(ev3_path)
        ev3_lp = np.log1p(d["predict"].to_numpy()[np.argsort(d["user_id"].to_numpy())])
        ts3 = pl.read_csv("submissions/two_stage_v3.csv")
        ts3_lp = np.log1p(ts3["predict"].to_numpy()[np.argsort(ts3["user_id"].to_numpy())])
        for name, pred_lb in preds_lb.items():
            pred_lb = np.clip(np.nan_to_num(pred_lb), 0, None)
            pred_lp = np.log1p(pred_lb) - np.log1p(pred_lb).mean() + EL
            c_ev3 = np.corrcoef(pred_lp, ev3_lp)[0, 1]
            c_ts3 = np.corrcoef(pred_lp, ts3_lp)[0, 1]
            print(f"  LB corr: {name}: corr(ev3)={c_ev3:.5f}  corr(ts3)={c_ts3:.5f}")
else:
    print(f"Якорь {LB_ANCHOR} не найден в BT, пропускаем LB submission.")

import os; os._exit(0)
