#!/usr/bin/env bash
# Re-run the two architectures whose first implementation was defective (NODE collapsed to a
# constant, TabM's ensemble members were initialised too close together), plus NGBoost which was
# killed mid-run. Waits for the current driver so the GPU is not shared.
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache ECUP_FS=/root/ecup/fs ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23

{
  echo "[zoo3] waiting for driver_zoo2 ..."
  until grep -q 'ALL DONE' model_out/driver_zoo2.log 2>/dev/null; do sleep 30; done
  echo "[zoo3] driver_zoo2 finished"

  for m in zoo_node zoo_tabm; do
    echo "[zoo3] === $m (исправленная реализация) ==="
    .venv/bin/python -u models/t17_tabnn_zoo.py "$m" > "model_out/fix_$m.log" 2>&1
    grep -E 'probe 2026|-> zoo|FAILED' "model_out/fix_$m.log" | tail -3
  done

  echo "[zoo3] === NGBoost ==="
  .venv/bin/python -u models/t19_distributional.py > model_out/t19_distributional.log 2>&1
  grep -E 'probe 2026|-> zoo|FAILED' model_out/t19_distributional.log | tail -8
  echo "[zoo3] ALL DONE"
} > model_out/driver_zoo3.log 2>&1
