#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[q2] === q1b: честные факторы + два контроля ==="
  .venv/bin/python -u exp/q1b_factors_honest.py > model_out/q1b.log 2>&1
  tail -22 model_out/q1b.log
  echo "[q2] === q4b: перенос сегментных весов между якорями ==="
  .venv/bin/python -u exp/q4b_segstack_transfer.py > model_out/q4b.log 2>&1
  tail -16 model_out/q4b.log
  echo "[q2] === q2: дневная генеративная (исправлено) ==="
  .venv/bin/python -u exp/q2_daily_gen.py > model_out/q2.log 2>&1
  tail -22 model_out/q2.log
  echo "[q2] ALL DONE"
} > model_out/driver_q2.log 2>&1
