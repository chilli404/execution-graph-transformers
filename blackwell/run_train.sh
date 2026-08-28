#!/bin/bash
# Gate 3: Full 430M training (polymorphic + sequential specialist)
# Run with: tsp bash blackwell/run_train.sh
#
# Queues two sequential tsp jobs (second depends on first via -D):
#   1. Polymorphic model (graph-consistent): 12000 steps
#   2. Sequential specialist (baseline): 12000 steps
#
# Expected time: 6-12 hours each depending on GPU throughput
# Expected tokens: 12000 steps * 8 batch * 1024 ctx = ~98M tokens
#
# Monitor: tail -f runs/430m_poly_full/train_log.jsonl
#          tsp -t (show running task output)
set -euo pipefail
cd "$(dirname "$0")/.."

PROJ_DIR="$(pwd)"

echo "=== 430M Full Training ==="
echo "$(date): Queuing polymorphic and sequential training runs"
echo "Project dir: $PROJ_DIR"

# Job 1: Polymorphic (graph-consistent) model
JOB1=$(tsp bash -c "
cd $PROJ_DIR
echo \"\$(date): Starting 430M polymorphic training\"
uv run python -m fogen.training.train \
    --config blackwell/configs/scale430m_poly_full.yaml \
    --seed 42 \
    --out runs/430m_poly_full \
    --mlflow --no-wandb
echo \"\$(date): Polymorphic training complete\"
")

echo "Queued polymorphic training as job $JOB1"

# Job 2: Sequential specialist (depends on Job 1 to avoid GPU contention)
JOB2=$(tsp -D $JOB1 bash -c "
cd $PROJ_DIR
echo \"\$(date): Starting 430M sequential specialist training\"
uv run python -m fogen.training.train \
    --config blackwell/configs/scale430m_seq_full.yaml \
    --seed 42 \
    --out runs/430m_seq_full \
    --mlflow --no-wandb
echo \"\$(date): Sequential training complete\"
")

echo "Queued sequential training as job $JOB2 (depends on $JOB1)"
echo ""
echo "Monitor:"
echo "  tsp              # job status"
echo "  tsp -t           # running task output"
echo "  tail -f runs/430m_poly_full/train_log.jsonl"
