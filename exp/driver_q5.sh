#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[q5] === q2b: дневная генеративная + frailty + финальный вектор ==="
  .venv/bin/python -u exp/q2b_daily_frailty.py > model_out/q2b.log 2>&1
  tail -30 model_out/q2b.log
  echo "[q5] ALL DONE"
} > model_out/driver_q5.log 2>&1
