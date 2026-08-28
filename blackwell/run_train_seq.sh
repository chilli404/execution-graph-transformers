#!/bin/bash
# Gate 3: Full 430M sequential specialist (12000 steps)
# Queue after poly finishes: tsp bash blackwell/run_train_seq.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "$(date): Starting 430M sequential specialist training"
uv run python -u -m fogen.training.train \
    --config blackwell/configs/scale430m_seq_full.yaml \
    --seed 42 \
    --out runs/430m_seq_full \
    --mlflow --no-wandb
echo "$(date): Sequential training complete"
