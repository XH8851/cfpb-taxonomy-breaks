#!/bin/bash
# Phase 3 GPU work, run ALONE (after the CPU tier queue and the D-period freeze have finished),
# because ModernBERT training plus CPU jobs exhausted the 16 GB unified memory on 2026-10-31.
# Order: ModernBERT A, B -> LLM few-shot A, B -> contamination probe.
# Progress: tail -f results/logs/run_phase3_gpu.log
P="${ROOT:-$(cd "$(dirname "$0")/.." \&\& pwd)}"
PY="$P/.venv/bin/python -I -u"
M=mlx-community/Qwen3-4B-Instruct-2507-4bit
cd /tmp
for f in run_phase3_cpu1 run_phase3_freeze; do
  while kill -0 $(cat $P/results/logs/$f.pid) 2>/dev/null; do sleep 60; done
done
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
set -e
for W in A B; do
  echo "[$(date)] mbert $W start"
  $PY $P/scripts/06v1_modernbert.py $P/data $P/results $W > $P/results/logs/06v1_mbert_$W.log 2>&1
  echo "[$(date)] mbert $W done"
done
for W in A B; do
  echo "[$(date)] LLM $W start"
  $PY $P/scripts/08_llm_tier.py --model $M --items $P/results/v1/llm_items_$W.parquet --submap $P/data/submap_v1.json \
      --out $P/results/v1/llm_scores_$W.parquet --fewshot $P/results/v1/llm_pool_$W.parquet --k 4 > $P/results/logs/08v1_llm_$W.log 2>&1
  echo "[$(date)] LLM $W done"
done
$PY $P/scripts/08_llm_tier.py --model $M --items $P/results/v1/contam_items.parquet --submap $P/data/submap_v1.json \
    --out $P/results/v1/contam_scores.parquet > $P/results/logs/08v1_contam.log 2>&1
echo "[$(date)] contamination probe done"
echo "[$(date)] ALL DONE"
