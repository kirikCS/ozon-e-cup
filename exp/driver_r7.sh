#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[r7] жду r6 ..."
  until grep -q "ALL DONE" model_out/driver_r6.log 2>/dev/null; do sleep 30; done
  echo "[r7] === r3b: сдвиги, честная версия ==="
  .venv/bin/python -u exp/r3b_timeshift_honest.py > model_out/r3b.log 2>&1
  grep -vE "Warning|warn" model_out/r3b.log | tail -26
  echo "[r7] ALL DONE"
} > model_out/driver_r7.log 2>&1
