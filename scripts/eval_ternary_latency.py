"""Measure latency under ternary execution masks.

Profiles wall-clock time for sequential, parallel, and various skip
configurations. Shows the speed/quality tradeoff for ternary programs.

Usage:
  python scripts/eval_ternary_latency.py \
      --ckpt runs/430m_ternary_poly/ckpts/step012000.safetensors \
      --config configs/scale430m_ternary_polymorphic.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/ternary_latency.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader
from fogen.model import GPT, ModelConfig


def time_forward(model, x, mode, warmup=3, repeats=20):
    for _ in range(warmup):
        model(x, mode=mode)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x, mode=mode)
        if x.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = GPT(mcfg).to(device=device, dtype=param_dtype)
    state = {k: v.to(param_dtype) for k, v in load_file(args.ckpt).items()}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("rope_") for k in missing)
    model.eval()

    loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=42, device=device)
    x, _ = loader.next_batch()
    n_layers = mcfg.n_layer

    results = []

    # Baseline: all sequential
    t_seq = time_forward(model, x, "sequential")
    results.append({"mode": "all_sequential", "n_skip": 0, "n_parallel": 0,
                     "latency_ms": t_seq * 1000, "speedup": 1.0})
    print(f"  all_sequential: {t_seq*1000:.2f}ms (baseline)")

    # All parallel
    t_par = time_forward(model, x, "parallel")
    results.append({"mode": "all_parallel", "n_skip": 0, "n_parallel": n_layers,
                     "latency_ms": t_par * 1000, "speedup": t_seq / t_par})
    print(f"  all_parallel:   {t_par*1000:.2f}ms ({t_seq/t_par:.2f}x)")

    # All parallel fused
    t_fused = time_forward(model, x, "parallel_fused")
    results.append({"mode": "all_fused", "n_skip": 0, "n_parallel": n_layers,
                     "latency_ms": t_fused * 1000, "speedup": t_seq / t_fused})
    print(f"  all_fused:      {t_fused*1000:.2f}ms ({t_seq/t_fused:.2f}x)")

    # Skip N layers (best-first: skip layers with smallest contribution)
    # Use sequential as base, skip from the end (last layers tend to contribute less)
    for n_skip in range(1, min(n_layers, 11)):
        mask = ["sequential"] * n_layers
        for i in range(n_layers - n_skip, n_layers):
            mask[i] = "skip"
        t = time_forward(model, x, mask)
        results.append({"mode": f"skip_{n_skip}_last", "n_skip": n_skip,
                         "n_parallel": 0, "latency_ms": t * 1000,
                         "speedup": t_seq / t})
        print(f"  skip {n_skip:>2} (last): {t*1000:.2f}ms ({t_seq/t:.2f}x)")

    # Mixed: parallel + skip
    for n_skip in [2, 4, 6]:
        mask = ["parallel"] * n_layers
        for i in range(n_layers - n_skip, n_layers):
            mask[i] = "skip"
        t = time_forward(model, x, mask)
        results.append({"mode": f"parallel_skip_{n_skip}", "n_skip": n_skip,
                         "n_parallel": n_layers - n_skip,
                         "latency_ms": t * 1000, "speedup": t_seq / t})
        print(f"  par+skip {n_skip:>2}:   {t*1000:.2f}ms ({t_seq/t:.2f}x)")

    output = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "batch_size": 8,
        "ctx_len": mcfg.ctx_len,
        "baseline_ms": t_seq * 1000,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
