#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () { echo "[r] === $1 ==="; .venv/bin/python -u "exp/$2" > "model_out/$3.log" 2>&1; tail -n "$4" "model_out/$3.log"; }
{
  run "r1: цель как коэффициент к прошлому месяцу" r1_ratio_target.py r1 26
  run "r5: цена клика"                              r5_click_value.py  r5 24
  run "r4: канал каталог/поиск — ICC и признаки"    r4_channel_stats.py r4 30
  run "r2: асимметричное разделение пайплайна"      r2_pipeline_split.py r2 20
  run "r3: сдвиги по таймлайну"                     r3_timeshift.py     r3 26
  run "q2c: дневная генеративная + квадратура"      q2c_daily_quad.py   q2c 26
  echo "[r] ALL DONE"
} > model_out/driver_r.log 2>&1
