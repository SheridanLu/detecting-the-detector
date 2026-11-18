#!/bin/bash
# Follow-up GPU run (arms C/D, Qwen + RoBERTa) followed by UNCONDITIONAL production restore (own snapshot + own log).
R=/data/int4096_unified_v4
export SNAP=$R/prod_snapshot_2 RLOG=$R/logs/restore_followup.log PIDDIR=$R/logs/gpu_followup
echo "RUN START $(date -u +'%F %T UTC')" > $R/logs/run_followup.log
$R/venv/bin/python $R/scripts/run_gpu_followup.py >> $R/logs/run_followup.log 2>&1
echo "ORCH EXIT=$? $(date -u +'%F %T UTC')" >> $R/logs/run_followup.log
bash $R/scripts/restore_prod.sh
echo "ALL_DONE $(date -u +"%F %T UTC")" >> $R/logs/run_followup.log
cd $R; export PIDDIR=
nice -n 19 venv/bin/python scripts/followup_metrics_gpu.py roberta 5 > logs/followup_metrics_roberta.log 2>&1
nice -n 19 venv/bin/python scripts/followup_metrics_gpu.py qwen 2 > logs/followup_metrics_qwen.log 2>&1
echo "METRICS_DONE $(date -u +"%F %T UTC")" >> $R/logs/run_followup.log
