#!/bin/bash
# Waits for the Product T0 and F tiers, then freezes CoD knobs on the dev period D (17_freeze_dev.py).
P="${ROOT:-$(cd "$(dirname "$0")/.." \&\& pwd)}"
while [ ! -f $P/results/v1/tier_tfidf_product.parquet ] || [ ! -f $P/results/v1/tier_emb_product.parquet ]; do sleep 120; done
sleep 30
echo "[$(date)] freeze start"
cd /tmp && $P/.venv/bin/python -I -u $P/scripts/17_freeze_dev.py $P/results > $P/results/logs/17_freeze.log 2>&1
echo "[$(date)] freeze done (exit $?)"
