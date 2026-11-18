#!/bin/bash
# GPU run followed by unconditional production restore.
R=/data/int4096_unified_v4
echo "RUN START $(date -u +'%F %T UTC')" > $R/logs/run_all.log
$R/venv/bin/python $R/scripts/run_gpu.py >> $R/logs/run_all.log 2>&1
echo "ORCH EXIT=$? $(date -u +'%F %T UTC')" >> $R/logs/run_all.log
bash $R/scripts/restore_prod.sh
echo "ALL_DONE $(date -u +'%F %T UTC')" >> $R/logs/run_all.log
