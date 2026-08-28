#!/bin/bash
# Gate 1: Memory + numerical smoke test (20 steps, ~2 minutes)
# Run with: tsp bash blackwell/run_smoke.sh
#
# SUCCESS: no OOM, no NaN, loss printed for 20 steps
# FAILURE: OOM → reduce batch_seqs to 4 in the config
#          NaN → check LR or model init
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Gate 1: 430M Polymorphic Smoke Test ==="
echo "$(date): Starting 20-step smoke test"

uv run python -m fogen.training.train \
    --config blackwell/configs/scale430m_poly_smoke.yaml \
    --seed 42 \
    --out runs/430m_poly_smoke \
    --mlflow --no-wandb

echo ""
echo "=== SMOKE TEST COMPLETE ==="
echo "Check runs/430m_poly_smoke/train_log.jsonl for loss values"
echo "If loss decreased and no NaN, proceed with: tsp bash blackwell/run_train.sh"
tail -3 runs/430m_poly_smoke/train_log.jsonl
