"""End-to-end ternary compiler evaluation.

For each skip budget (0, 1, 2, ... n_layers//2):
1. Measure per-layer skip cost (single-layer skip ΔBPB)
2. Greedy selection: skip cheapest layers first
3. Profile actual wall-clock latency for the compiler-selected mask
4. Measure actual quality (BPB) for that exact mask

Produces one coherent table: compiler mask → measured latency → measured quality.

Usage:
  python scripts/eval_ternary_compiler_e2e.py \
      --ckpt runs/430m_ternary_gradnorm/ckpts/step012000.safetensors \
      --config configs/scale430m_ternary_gradnorm.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/ternary_compiler_e2e_430m.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader
from fogen.model import GPT, ModelConfig


def evaluate_bpb(model, loader, mode, n_batches=16):
    """Measure BPB under a given execution mode/mask."""
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = loader.next_batch()
        with torch.no_grad(), torch.autocast(
            device_type=x.device.type, dtype=torch.bfloat16,
            enabled=x.device.type == "cuda",
        ):
            logits = model(x, mode=mode)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    ce = total_loss / total_tokens
    return ce / np.log(2)


def time_forward(model, x, mode, warmup=3, repeats=20):
    """Measure wall-clock latency for a forward pass."""
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
    parser.add_argument("--n_eval_batches", type=int, default=16)
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

    n_layers = mcfg.n_layer
    cal_loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=42, device=device)
    eval_loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=999, device=device)

    # Step 1: Measure per-layer skip cost (calibration set)
    print("=== Step 1: Per-layer skip costs (calibration seed=42) ===")
    baseline_bpb = evaluate_bpb(model, cal_loader, "sequential", args.n_eval_batches)
    print(f"  Baseline (all seq) BPB: {baseline_bpb:.4f}")

    skip_costs = []
    for layer in range(n_layers):
        mask = ["sequential"] * n_layers
        mask[layer] = "skip"
        bpb = evaluate_bpb(model, cal_loader, mask, args.n_eval_batches)
        cost = bpb - baseline_bpb
        skip_costs.append(cost)
        print(f"  Layer {layer:>2}: ΔBPB = {cost:+.4f}")

    skip_costs = np.array(skip_costs)

    # Step 2: Greedy order (cheapest first)
    greedy_order = np.argsort(skip_costs).tolist()
    print(f"\n  Greedy order: {greedy_order}")

    # Step 3: For each skip budget, build compiler mask and evaluate on HELD-OUT data
    print("\n=== Step 2: Compiler Pareto frontier (eval seed=999) ===")

    x_timing, _ = eval_loader.next_batch()

    # Baseline: all sequential (eval set)
    baseline_bpb_eval = evaluate_bpb(model, eval_loader, "sequential", args.n_eval_batches)
    t_seq = time_forward(model, x_timing, "sequential")
    print(f"  Baseline (all seq): {baseline_bpb_eval:.4f} BPB, {t_seq*1000:.1f}ms")

    # Stage 1: all parallel fused (should be nearly free)
    fused_bpb = evaluate_bpb(model, eval_loader, "parallel_fused", args.n_eval_batches)
    fused_delta = fused_bpb - baseline_bpb_eval
    t_fused = time_forward(model, x_timing, "parallel_fused")
    fused_speedup = t_seq / t_fused
    print(f"  Stage 1 (all par):  {fused_bpb:.4f} BPB (ΔBPB={fused_delta:+.4f}), "
          f"{t_fused*1000:.1f}ms ({fused_speedup:.2f}x)")

    # Stage 2: parallel + greedy skip
    results = []
    print(f"\n  Stage 2: parallel + greedy skip (cheapest layers first)")
    print(f"  {'n_skip':>6} {'Layers skipped':>20} {'Pred ΔBPB':>10} "
          f"{'Actual ΔBPB':>12} {'Latency':>10} {'Speedup':>8}")
    print("  " + "-" * 72)

    for n_skip in range(0, min(n_layers // 2 + 1, 11)):
        selected = sorted(greedy_order[:n_skip])
        predicted_cost = float(skip_costs[selected].sum()) if selected else 0.0

        # Build mask: all parallel + selected skips
        compiler_mask = ["parallel"] * n_layers
        for i in selected:
            compiler_mask[i] = "skip"

        # Measure actual quality on held-out data
        actual_bpb = evaluate_bpb(model, eval_loader, compiler_mask, args.n_eval_batches)
        actual_delta = actual_bpb - baseline_bpb_eval

        # Measure actual latency
        latency = time_forward(model, x_timing, compiler_mask)
        speedup = t_seq / latency

        selected_str = str(selected) if selected else "[]"
        print(f"  {n_skip:>6} {selected_str:>20} {predicted_cost:>10.4f} "
              f"{actual_delta:>12.4f} {latency*1000:>9.1f}ms {speedup:>7.2f}x")

        results.append({
            "n_skip": n_skip,
            "selected_layers": selected,
            "predicted_delta_bpb": round(predicted_cost, 5),
            "actual_delta_bpb": round(actual_delta, 5),
            "actual_bpb": round(actual_bpb, 5),
            "latency_ms": round(latency * 1000, 2),
            "speedup": round(speedup, 3),
        })

    output = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "baseline_bpb": round(baseline_bpb, 5),
        "baseline_latency_ms": round(t_seq * 1000, 2),
        "fused_bpb": round(fused_bpb, 5),
        "fused_delta_bpb": round(fused_delta, 5),
        "fused_latency_ms": round(t_fused * 1000, 2),
        "fused_speedup": round(t_seq / t_fused, 3),
        "skip_costs": [round(c, 5) for c in skip_costs.tolist()],
        "greedy_order": greedy_order,
        "compiler_results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
