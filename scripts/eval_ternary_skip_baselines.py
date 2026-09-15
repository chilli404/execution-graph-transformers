"""Compare skip strategies: greedy vs naive baselines.

For each skip count, measures quality (held-out BPB) and latency for:
  - Greedy: skip cheapest layers first (compiler output)
  - First N: skip layers 0, 1, 2, ...
  - Last N: skip layers L-1, L-2, ...
  - Middle N: skip layers centered around L/2
  - Evens: skip even-indexed layers first
  - Random: average over 10 random skip selections

Uses the same calibration/eval split as eval_ternary_compiler_e2e.py.

Usage:
  python scripts/eval_ternary_skip_baselines.py \
      --ckpt runs/430m_ternary_gradnorm/ckpts/step012000.safetensors \
      --config configs/scale430m_ternary_gradnorm.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/ternary_skip_baselines_430m.json
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
    return (total_loss / total_tokens) / np.log(2)


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


def build_mask(n_layers, skip_indices):
    mask = ["parallel_fused"] * n_layers
    for i in skip_indices:
        mask[i] = "skip"
    return mask


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

    # Calibration: per-layer skip costs
    print("Measuring per-layer skip costs (calibration, seed=42)...")
    cal_baseline = evaluate_bpb(model, cal_loader, "sequential", args.n_eval_batches)
    skip_costs = []
    for layer in range(n_layers):
        mask = ["sequential"] * n_layers
        mask[layer] = "skip"
        bpb = evaluate_bpb(model, cal_loader, mask, args.n_eval_batches)
        skip_costs.append(bpb - cal_baseline)
    skip_costs = np.array(skip_costs)
    greedy_order = np.argsort(skip_costs).tolist()

    # Eval baseline
    seq_bpb = evaluate_bpb(model, eval_loader, "sequential", args.n_eval_batches)
    x_timing, _ = eval_loader.next_batch()
    t_seq = time_forward(model, x_timing, "sequential")

    rng = np.random.RandomState(123)

    skip_counts = [1, 2, 3, 4, 5, 6, 8, 10]
    skip_counts = [k for k in skip_counts if k <= n_layers // 2]

    strategies = {
        "greedy": lambda k: sorted(greedy_order[:k]),
        "first_n": lambda k: list(range(k)),
        "last_n": lambda k: list(range(n_layers - k, n_layers)),
        "middle_n": lambda k: sorted(list(range(n_layers // 2 - k // 2, n_layers // 2 - k // 2 + k))),
        "evens_first": lambda k: sorted([i for i in range(0, n_layers, 2)][:k]),
    }

    print(f"\n{'Strategy':<15} {'n_skip':>6} {'ΔBPB':>8} {'Speedup':>8}")
    print("-" * 42)

    all_results = []
    for k in skip_counts:
        row = {"n_skip": k}
        for name, selector in strategies.items():
            indices = selector(k)
            mask = build_mask(n_layers, indices)
            bpb = evaluate_bpb(model, eval_loader, mask, args.n_eval_batches)
            delta = bpb - seq_bpb
            latency = time_forward(model, x_timing, mask)
            speedup = t_seq / latency
            row[name] = {
                "indices": indices,
                "delta_bpb": round(delta, 5),
                "bpb": round(bpb, 5),
                "latency_ms": round(latency * 1000, 2),
                "speedup": round(speedup, 3),
            }
            print(f"{name:<15} {k:>6} {delta:>+8.4f} {speedup:>7.2f}x")

        # Random: average over 10 draws
        random_deltas = []
        for _ in range(10):
            indices = sorted(rng.choice(n_layers, k, replace=False).tolist())
            mask = build_mask(n_layers, indices)
            bpb = evaluate_bpb(model, eval_loader, mask, args.n_eval_batches)
            random_deltas.append(bpb - seq_bpb)
        row["random"] = {
            "mean_delta_bpb": round(float(np.mean(random_deltas)), 5),
            "std_delta_bpb": round(float(np.std(random_deltas)), 5),
        }
        print(f"{'random (10x)':<15} {k:>6} {np.mean(random_deltas):>+8.4f}")
        print()

        all_results.append(row)

    # Summary: greedy vs best naive at each k
    print("=" * 50)
    print(f"{'n_skip':>6} {'Greedy':>10} {'Best naive':>12} {'Greedy wins?':>14}")
    print("-" * 50)
    for row in all_results:
        k = row["n_skip"]
        greedy_d = row["greedy"]["delta_bpb"]
        naive_best = min(row[s]["delta_bpb"] for s in strategies if s != "greedy")
        wins = "YES" if greedy_d <= naive_best else "no"
        print(f"{k:>6} {greedy_d:>+10.4f} {naive_best:>+12.4f} {wins:>14}")

    output = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "seq_bpb": round(seq_bpb, 5),
        "seq_latency_ms": round(t_seq * 1000, 2),
        "skip_costs": [round(c, 5) for c in skip_costs.tolist()],
        "greedy_order": greedy_order,
        "results": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
