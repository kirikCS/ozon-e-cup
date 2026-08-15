#!/usr/bin/env bash
# Re-run the seven architectures that were killed by the NODE device-side assert, each in its own
# process so a CUDA failure in one cannot poison the others. Then the special heads.
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache ECUP_FS=/root/ecup/fs ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23

{
  for m in zoo_node zoo_tabr zoo_modernnca zoo_tabm zoo_excelformer zoo_carte zoo_tabnet; do
    echo "[zoo2] === $m ==="
    .venv/bin/python -u models/t17_tabnn_zoo.py "$m" > "model_out/zoo_$m.log" 2>&1
    echo "[zoo2] $m exit=$?"
    grep -E 'probe 2026|-> zoo|FAILED' "model_out/zoo_$m.log" | tail -4
  done
  echo "[zoo2] === t18_heads ==="
  .venv/bin/python -u models/t18_heads.py > model_out/t18_heads.log 2>&1
  echo "[zoo2] t18 exit=$?"
  grep -E 'probe 2026|-> zoo|FAILED' model_out/t18_heads.log | tail -8
  echo "[zoo2] ALL DONE"
} > model_out/driver_zoo2.log 2>&1
