#!/usr/bin/env bash
cd /root/ecup
export PYTHONPATH=code ECUP_CACHE=cache ECUP_OUT=model_out OMP_NUM_THREADS=11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "[r6] жду завершения предыдущей очереди ..."
  until grep -q "ALL DONE" model_out/driver_r.log 2>/dev/null; do sleep 30; done
  echo "[r6] === r6: r1 и r5 только на тёплых, бленд вместо маршрутизации ==="
  .venv/bin/python -u exp/r6_warm_only.py > model_out/r6.log 2>&1
  grep -vE "Warning|warn" model_out/r6.log | tail -46
  echo "[r6] ALL DONE"
} > model_out/driver_r6.log 2>&1
