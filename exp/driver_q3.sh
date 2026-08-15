#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[q3] === q2: дневная генеративная модель на GPU ==="
  .venv/bin/python -u exp/q2_daily_gen.py > model_out/q2.log 2>&1
  tail -22 model_out/q2.log
  echo "[q3] === q4c: сегментные веса с seq_pure в базисе ==="
  .venv/bin/python -u exp/q4c_segstack_seq.py > model_out/q4c.log 2>&1
  tail -24 model_out/q4c.log
  echo "[q3] ALL DONE"
} > model_out/driver_q3.log 2>&1
