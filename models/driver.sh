#!/usr/bin/env bash
# Server-side driver: survives ssh drops. Waits for Tier 5, then runs Tier 6 and the summary.
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache
export ECUP_FS=/root/ecup/fs
export ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23

{
  echo "[driver] waiting for Tier 5 ..."
  until grep -q 'Tier 5 done' model_out/t05_sequence.log 2>/dev/null; do sleep 30; done
  echo "[driver] Tier 5 done"
  grep -E 'RMSLE|FAILED' model_out/t05_sequence.log

  echo "[driver] Tier 6 ..."
  python3 models/clean_sb.py model_out/scoreboard.jsonl 'Blend:' 'Isotonic'
  .venv/bin/python -u models/t06_ensemble.py

  echo "[driver] finalize ..."
  .venv/bin/python models/finalize.py
  .venv/bin/python models/summarize.py model_out
  echo "[driver] ALL DONE"
} > model_out/driver.log 2>&1
