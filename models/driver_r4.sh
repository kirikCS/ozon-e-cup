#!/usr/bin/env bash
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache ECUP_FS=/root/ecup/fs ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23
{
  echo "[r4] === ElasticNet как базовая модель ==="
  .venv/bin/python -u models/t26_elastic_base.py > model_out/t26_elastic.log 2>&1
  grep -E 'probe|выбрано|ненулевых|-> |FAILED|Error' model_out/t26_elastic.log | tail -20
  echo "[r4] === OOF скрытые представления ==="
  .venv/bin/python -u models/t25_oof_meta.py > model_out/t25_oof.log 2>&1
  grep -E 'probe|LOO|-> |FAILED|Error' model_out/t25_oof.log | tail -20
  echo "[r4] ALL DONE"
} > model_out/driver_r4.log 2>&1
