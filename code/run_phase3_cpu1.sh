#!/bin/bash
# Phase 3 CPU queue (detached, one worker to leave memory for the GPU job): monthly tiers.
# Progress: tail -f results/logs/run_phase3_cpu1.log ; per-job logs results/logs/14_<feat>_<task>.log
set -e
P="${ROOT:-$(cd "$(dirname "$0")/.." \&\& pwd)}"
cd /tmp
for job in "emb product" "tfidf issue" "emb issue"; do
  set -- $job
  echo "[$(date)] $1 $2 start"
  $P/.venv/bin/python -I -u $P/scripts/14_monthly_tiers.py $P/data $P/results $1 $2 2 > $P/results/logs/14_$1_$2.log 2>&1
  echo "[$(date)] $1 $2 done"
done
echo "[$(date)] ALL DONE"
