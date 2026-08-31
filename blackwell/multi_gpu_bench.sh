#!/bin/bash
# Multi-GPU tensor parallelism benchmark
#
# Run on a fresh 2x or 4x H100/A100 instance.
# Uploads the checkpoint, installs everything, benchmarks all modes.
#
# Usage:
#   1. Rent a 2x+ GPU machine (Lambda, RunPod, etc.)
#   2. scp this script + the HF checkpoint to the machine:
#        scp -r blackwell/multi_gpu_bench.sh runs/430m_poly_hf user@host:~/
#        scp -r runs/7b_poly_hf user@host:~/   # optional, for 7B
#   3. ssh user@host
#   4. bash multi_gpu_bench.sh 430m_poly_hf 2   # model_dir, tp_size
#
# Results saved to bench_tp_results.json
set -euo pipefail

MODEL_DIR="${1:?Usage: bash multi_gpu_bench.sh <hf_model_dir> <tp_size>}"
TP="${2:?Specify tensor parallel size (2 or 4)}"
PORT=8000
NUM_PROMPTS=100
OUTPUT="bench_tp${TP}_results.json"

echo "=== Multi-GPU Serving Benchmark ==="
echo "Model: $MODEL_DIR"
echo "Tensor parallel: $TP"
echo "GPUs:"
nvidia-smi -L

# Install if needed
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv-bench ]; then
    uv venv .venv-bench --python 3.12
    .venv-bench/bin/pip install "vllm>=0.28" torch
fi
PYTHON=".venv-bench/bin/python"
VLLM=".venv-bench/bin/vllm"

# Also need transformers for the model config
.venv-bench/bin/pip install transformers tokenizers safetensors 2>/dev/null

wait_for_server() {
    local timeout=300
    local start=$(date +%s)
    while true; do
        if curl -fs http://localhost:$PORT/health >/dev/null 2>&1; then
            echo "  Server ready ($(($(date +%s) - start))s)"
            return 0
        fi
        if [ $(($(date +%s) - start)) -ge $timeout ]; then
            echo "  Server failed to start"
            return 1
        fi
        sleep 3
    done
}

bench_mode() {
    local MODE=$1
    echo ""
    echo "============================================================"
    echo "Mode: $MODE | TP=$TP"
    echo "============================================================"

    # Start server
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --trust-remote-code \
        --dtype bfloat16 \
        --port $PORT \
        --tensor-parallel-size "$TP" \
        --hf-overrides "{\"execution_mode\":\"$MODE\"}" \
        >/tmp/vllm_server.log 2>&1 &
    local PID=$!

    if ! wait_for_server; then
        kill $PID 2>/dev/null; wait $PID 2>/dev/null
        echo "  FAILED"
        return
    fi

    for CONC in 1 8 64; do
        for ILEN in 128 512; do
            for OLEN in 32 128; do
                local N=$NUM_PROMPTS
                [ $CONC -eq 1 ] && N=50

                echo ""
                echo "  C=$CONC, input=$ILEN, output=$OLEN, prompts=$N"
                $VLLM bench serve \
                    --backend vllm \
                    --model "$MODEL_DIR" \
                    --port $PORT \
                    --num-prompts $N \
                    --request-rate inf \
                    --max-concurrency $CONC \
                    --random-input-len $ILEN \
                    --random-output-len $OLEN \
                    --temperature 0 2>&1 | grep -E "throughput|TTFT|TPOT|ITL|duration"
            done
        done
    done

    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "  Server stopped"
    sleep 5
}

# Run all three modes
bench_mode sequential | tee -a /tmp/bench_log.txt
bench_mode parallel | tee -a /tmp/bench_log.txt
bench_mode parallel_fused | tee -a /tmp/bench_log.txt

echo ""
echo "=== Done ==="
echo "Full log: /tmp/bench_log.txt"
echo "Copy results back: scp user@host:/tmp/bench_log.txt ."
