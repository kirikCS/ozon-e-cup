#!/usr/bin/env bash
# Launch a tier script detached on the server. Usage: bash models/run.sh t03_gbdt
set -e
cd /root/ecup
export ECUP_CACHE=/root/ecup/cache
export ECUP_FS=/root/ecup/fs
export ECUP_OUT=/root/ecup/model_out
export OMP_NUM_THREADS=23
mkdir -p model_out fs
setsid nohup .venv/bin/python -u "models/$1.py" > "model_out/$1.log" 2>&1 < /dev/null &
echo "launched $1 pid $!"
