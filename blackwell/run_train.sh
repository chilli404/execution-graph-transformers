#!/bin/bash
# Gate 3: Full 430M polymorphic training (12000 steps)
# Queue with: tsp bash blackwell/run_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "$(date): Starting 430M polymorphic training"
uv run python -u -m fogen.training.train \
    --config blackwell/configs/scale430m_poly_full.yaml \
    --seed 42 \
    --out runs/430m_poly_full \
    --mlflow --no-wandb
echo "$(date): Polymorphic training complete"
