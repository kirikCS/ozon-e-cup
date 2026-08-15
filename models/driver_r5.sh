#!/usr/bin/env bash
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache ECUP_FS=/root/ecup/fs ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23
{
  echo "[r5] === панели каталожного канала ==="
  .venv/bin/python -u models/server_prep2.py 2>&1 | tail -8
  echo "[r5] === бустинг с v3 ==="
  .venv/bin/python -u models/t27_channel.py > model_out/t27_channel.log 2>&1
  grep -E 'probe |важности|топ:|-> |Error|Traceback' model_out/t27_channel.log | tail -20
  echo "[r5] ALL DONE"
} > model_out/driver_r5.log 2>&1
