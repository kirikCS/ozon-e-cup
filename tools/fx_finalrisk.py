"""Какие исторические кандидаты переживут переход public -> private.

Нулевой тест (code/fx_nulltest.py) дал кривую: подгонка по 50k публичных из точки,
где настоящего выигрыша нет, приносит на public чистым шумом
    ||z||_C = 0.005 -> +0.000094      private -0.000033
              0.010 -> +0.000167      private -0.000079
              0.020 -> +0.000287      private -0.000222
              0.027 -> +0.000313      private -0.000340
              0.035 -> +0.000316      private -0.000460
              0.050 -> +0.000360      private -0.000539
(0% разбиений из 40 улучшили private ни на одном радиусе)

Тест для любой пары: публичный выигрыш БОЛЬШЕ шумового потолка на своём радиусе -
значит выигрыш в основном настоящий и переживёт приват. Меньше - подгонка.

Usage: python -u code/fx_finalrisk.py > model_out/finalrisk.log 2>&1
"""
import json
import numpy as np
import polars as pl
from pathlib import Path

EL, VAR_L, R_KNN = 2.3284, 5.366802, 0.9518
REC_PATH = "submissions/kirill_stack_final_lb1646487.csv"
REC = 1.646487285085104

# кривая шумового потолка из нулевого теста, с поправкой sqrt(26/20)=1.14 на
# размерность (у нас 26 векторов, в симуляции было 20)
DIM_FIX = (26 / 20) ** 0.5
NULL_D = np.array([0.005, 0.010, 0.020, 0.027, 0.035, 0.050])
NULL_PUB = np.array([0.000094, 0.000167, 0.000287, 0.000313, 0.000316, 0.000360]) * DIM_FIX
NULL_PRIV = np.array([-0.000033, -0.000079, -0.000222, -0.000340, -0.000460, -0.000539]) * DIM_FIX

V24 = [
    ("bwn", "submissions/blend-with-neural.csv"),
    ("tcb", "submissions/TOTALCATBOOSTBLEND.csv"),
    ("nocal", "submissions/nocal_catboost.csv"),
    ("seq_pure", "submissions/seq_pure_9seed.csv"),
    ("stack_dl", "submissions/stack_dl.csv"),
    ("evgru2", "submissions/evgru2_final_lb1660490.csv"),
    ("evgru3", "submissions/evgru3_final_lb1656201.csv"),
    ("ts_emb", "submissions/ts_emb_9seed.csv"),
    ("ts_v3", "submissions/two_stage_v3.csv"),
    ("cb_v3", "submissions/catboost_v3.csv"),
    ("ts_9seed", "submissions/two_stage_btyd_9seed.csv"),
    ("ts_old", "submission/components/two_stage_btyd_shift.csv"),
    ("cb_old", "submission/components/catboost_shifted.csv"),
    ("seq_old", "submission/components/seq_hybrid.csv"),
    ("seq9", "submissions/seq_hybrid_9seed.csv"),
    ("dart", "submission/components/lgbm_dart.csv"),
    ("iso", "submission/components/nnls_isotonic_blend.csv"),
    ("knn", "submissions/knn_submit_lb.csv"),
    ("btyd_basic", "submissions/btyd_btyd_basic_lb.csv"),
    ("eb", "submission/components/eb_logblocks.csv"),
    ("knn_funnel", "submissions/knn_funnel_lb.csv"),
    ("knn_funnel2", "submissions/knn_funnel2_lb.csv"),
    ("funnel_lgbm", "submissions/funnel_lgbm_lb.csv"),
    ("daily", "release_blend/components/daily.csv"),
]


def load_lm(p):
    d = pl.read_csv(p)
    q = np.log1p(d["predict"].to_numpy()[np.argsort(d["user_id"].to_numpy())])
    return q - q.mean() + EL


M = np.stack([load_lm(p) for _, p in V24])
C = np.cov(M)
Am = np.concatenate([M, np.ones((1, M.shape[1]))], 0)
q_rec = load_lm(REC_PATH)
w_rec = np.linalg.lstsq(Am.T, q_rec, rcond=None)[0][:24]

CAND = [
    ("stack6 (безопасный финал)", "submission/stack6.csv", 1.6496637839),
    ("stack7_2s9seed", "submissions/stack7_2s9seed_lb1649381.csv", 1.649380786349335),
    ("stack_dl_fusion", "submissions/stack_dl_fusion_lb1648049.csv", 1.6480488273),
    ("stack13", "submissions/stack13_lb1647414.csv", 1.6474141758705303),
    ("stack_opt_blend", "submissions/stack_opt_blend_lb1647064.csv", 1.6470636973608355),
    ("kirill stack_opt_daily", "submissions/kirill_stack_opt_daily_lb1646812.csv", 1.6468118221271069),
    ("РЕКОРД Кирилла", REC_PATH, REC),
    ("SAFE_beat_record (мой)", "submissions/SAFE_beat_record.csv", None),
    ("B26_tight_D010 (мой)", "submissions/B26_tight_D010.csv", None),
]
PRED = {"SAFE_beat_record (мой)": 1.6462584, "B26_tight_D010 (мой)": 1.6463650}

print("=" * 96)
print("РИСК ПЕРЕОБУЧЕНИЯ ПОД PUBLIC: каждый кандидат против РЕКОРДА")
print("=" * 96)
print(f"{'кандидат':>26} {'LB':>11} {'||z||_C':>8} {'выигрыш':>10} "
      f"{'шум.потолок':>12} {'сигнал/шум':>11} {'вердикт':>16}")
rows = []
for tag, path, lb in CAND:
    if not Path(path).exists():
        print(f"{tag:>26}  НЕТ ФАЙЛА {path}")
        continue
    q = load_lm(path)
    w = np.linalg.lstsq(Am.T, q, rcond=None)[0][:24]
    z = w - w_rec
    d = float(np.sqrt(max(z @ C @ z, 0)))
    real_lb = lb if lb is not None else PRED[tag]
    gain = REC - real_lb                      # >0 = лучше рекорда на public
    nz = float(np.interp(d, NULL_D, NULL_PUB))
    npriv = float(np.interp(d, NULL_D, NULL_PRIV))
    snr = gain / nz if nz > 0 else float("inf")
    verd = ("настоящий" if snr > 2 else "спорный" if snr > 1 else
            "подгонка" if gain > 0 else "хуже рекорда")
    star = " (прогноз)" if lb is None else ""
    print(f"{tag:>26} {real_lb:11.7f} {d:8.4f} {gain:+10.6f} {nz:12.6f} "
          f"{snr:11.2f} {verd:>16}{star}")
    rows.append((tag, d, gain, nz, npriv, snr))

print()
print("сигнал/шум > 2  - выигрыш почти наверняка настоящий, переживёт приват")
print("сигнал/шум ~ 1  - неотличим от подгонки под публичные 50k")
print("отрицательный выигрыш - кандидат хуже рекорда уже на public")

print()
print("=" * 96)
print("ОЖИДАЕМЫЙ ЭФФЕКТ НА ПРИВАТЕ для тех, кто лучше рекорда на public")
print("=" * 96)
print(f"{'кандидат':>26} {'public':>10} {'штраф шума':>12} {'ожид. private':>14}")
for tag, d, gain, nz, npriv, snr in rows:
    if gain <= 0:
        continue
    # настоящая часть выигрыша = public выигрыш минус шумовая; штраф = кривая private
    exp_priv = gain + npriv if gain <= nz else gain - nz + abs(npriv) * 0
    print(f"{tag:>26} {gain:+10.6f} {npriv:+12.6f} {exp_priv:+14.6f}")
print()
print("Для кандидата, чей публичный выигрыш НИЖЕ шумового потолка, ожидаемый")
print("приватный эффект = public + (штраф из нулевого теста), то есть отрицательный.")
