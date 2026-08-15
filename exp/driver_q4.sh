#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[q4] жду завершения q4c ..."
  until grep -qE "^done|Traceback" model_out/q4c.log 2>/dev/null; do sleep 30; done
  echo "[q4] q4c завершён"
  tail -12 model_out/q4c.log
  echo "[q4] === q2: дневная генеративная, без GRU ==="
  .venv/bin/python -u exp/q2_daily_gen.py > model_out/q2.log 2>&1
  tail -22 model_out/q2.log
  echo "[q4] ALL DONE"
} > model_out/driver_q4.log 2>&1
