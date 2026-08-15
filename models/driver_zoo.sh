#!/usr/bin/env bash
# Runs the remaining zoo blocks in sequence after t17 finishes. Detached, so ssh drops don't matter.
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache ECUP_FS=/root/ecup/fs ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23

{
  echo "[zoo] waiting for t17 ..."
  until grep -qE '^done|Traceback' model_out/t17_tabnn_zoo.log 2>/dev/null; do sleep 30; done
  echo "[zoo] t17 finished"

  for s in t18_heads t20_tabpfn t19_distributional; do
    echo "[zoo] === $s ==="
    .venv/bin/python -u "models/$s.py" > "model_out/$s.log" 2>&1
    echo "[zoo] $s exit=$?"
    grep -E 'probe|-> zoo|FAILED' "model_out/$s.log" | tail -20
  done
  echo "[zoo] ALL DONE"
} > model_out/driver_zoo.log 2>&1
