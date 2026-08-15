#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
{
  echo "[q] === канальные панели (fx_panels) ==="
  .venv/bin/python -u code/fx_panels.py 2>&1 | tail -4
  echo "[q] === q1: спектр ко-активности (граф/кластеризация) ==="
  .venv/bin/python -u exp/q1_coactivity.py > model_out/q1.log 2>&1
  tail -22 model_out/q1.log
  echo "[q] === q4: сегментные веса стека, локальный скрин ==="
  .venv/bin/python -u exp/q4_segstack.py > model_out/q4.log 2>&1
  tail -30 model_out/q4.log
  echo "[q] === q3: канальная декомпозиция ==="
  .venv/bin/python -u exp/q3_twohead.py > model_out/q3.log 2>&1
  tail -18 model_out/q3.log
  echo "[q] === q2: дневная генеративная модель ==="
  .venv/bin/python -u exp/q2_daily_gen.py > model_out/q2.log 2>&1
  tail -20 model_out/q2.log
  echo "[q] ALL DONE"
} > model_out/driver_q.log 2>&1
