"""Compare layer-skip robustness: polymorphic vs sequential specialist.

Tests whether graph-consistency training confers robustness to a
graph rewrite type (layer skip) that was NOT part of training.

Usage:
  python scripts/eval_layer_skip.py \
      --poly_ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
      --seq_ckpt runs/430m_seq_full/ckpts/step012000.safetensors \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --seq_config blackwell/configs/scale430m_seq_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/layer_skip_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader, load_tokenizer
from fogen.evals.bpb import evaluate_bpb, val_stream
from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config_path, device):
    cfg = yaml.safe_load(open(config_path))
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(checkpoint).items()}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("rope_") for k in missing)
    return model.eval(), cfg, mcfg


def evaluate_mask(model, stream, token_bytes, mcfg, mask, device):
    old_mode = model.cfg.execution_mode
    model.cfg.execution_mode = mask
    result = evaluate_bpb(model, stream, token_bytes, mcfg.ctx_len,
                          batch_size=8, max_windows=64, device=str(device))
    model.cfg.execution_mode = old_mode
    return result["val_bpb"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poly_ckpt", required=True)
    parser.add_argument("--seq_ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq_config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading polymorphic model...", flush=True)
    poly_model, poly_cfg, mcfg = load_model(args.poly_ckpt, args.config, device)
    print("Loading sequential specialist...", flush=True)
    seq_model, seq_cfg, _ = load_model(args.seq_ckpt, args.seq_config, device)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    token_bytes = torch.tensor(
        [len(tokenizer.decode([i]).encode("utf-8")) for i in range(mcfg.vocab_size)],
        dtype=torch.float32)

    n_layers = mcfg.n_layer

    # Baseline BPB (no skips)
    print("\n=== Baselines ===", flush=True)
    poly_base = evaluate_mask(poly_model, stream, token_bytes, mcfg,
                              "sequential", device)
    seq_base = evaluate_mask(seq_model, stream, token_bytes, mcfg,
                             "sequential", device)
    print(f"  Poly baseline BPB:  {poly_base:.4f}")
    print(f"  Seq baseline BPB:   {seq_base:.4f}")

    # Single-layer skip
    print(f"\n=== Single-layer skip (L={n_layers}) ===", flush=True)
    single_results = []
    for skip_layer in range(n_layers):
        mask = ["sequential"] * n_layers
        mask[skip_layer] = "skip"

        poly_bpb = evaluate_mask(poly_model, stream, token_bytes, mcfg,
                                 mask, device)
        seq_bpb = evaluate_mask(seq_model, stream, token_bytes, mcfg,
                                mask, device)
        poly_deg = poly_bpb - poly_base
        seq_deg = seq_bpb - seq_base
        single_results.append({
            "layer": skip_layer,
            "poly_bpb": float(poly_bpb),
            "seq_bpb": float(seq_bpb),
            "poly_degradation": float(poly_deg),
            "seq_degradation": float(seq_deg),
        })
        print(f"  L{skip_layer:>2}: poly Δ={poly_deg:+.4f}  seq Δ={seq_deg:+.4f}  "
              f"{'poly better' if abs(poly_deg) < abs(seq_deg) else 'seq better'}",
              flush=True)

    # Multi-layer skip (ordered by single-layer degradation, skip most expendable first)
    print(f"\n=== Multi-layer skip ===", flush=True)
    poly_order = sorted(range(n_layers),
                        key=lambda i: single_results[i]["poly_degradation"])
    seq_order = sorted(range(n_layers),
                       key=lambda i: single_results[i]["seq_degradation"])

    multi_results = []
    for n_skip in [2, 4, 6, 8, 10]:
        if n_skip >= n_layers:
            break

        # Skip the n_skip least-impactful layers (by each model's own ranking)
        poly_mask = ["sequential"] * n_layers
        for i in poly_order[:n_skip]:
            poly_mask[i] = "skip"

        seq_mask = ["sequential"] * n_layers
        for i in seq_order[:n_skip]:
            seq_mask[i] = "skip"

        poly_bpb = evaluate_mask(poly_model, stream, token_bytes, mcfg,
                                 poly_mask, device)
        seq_bpb = evaluate_mask(seq_model, stream, token_bytes, mcfg,
                                seq_mask, device)

        poly_deg = poly_bpb - poly_base
        seq_deg = seq_bpb - seq_base
        multi_results.append({
            "n_skip": n_skip,
            "poly_skipped_layers": [i for i in poly_order[:n_skip]],
            "seq_skipped_layers": [i for i in seq_order[:n_skip]],
            "poly_bpb": float(poly_bpb),
            "seq_bpb": float(seq_bpb),
            "poly_degradation": float(poly_deg),
            "seq_degradation": float(seq_deg),
        })
        print(f"  Skip {n_skip:>2}: poly Δ={poly_deg:+.4f}  seq Δ={seq_deg:+.4f}",
              flush=True)

    # Summary
    poly_mean_single = np.mean([r["poly_degradation"] for r in single_results])
    seq_mean_single = np.mean([r["seq_degradation"] for r in single_results])
    poly_wins = sum(1 for r in single_results
                    if abs(r["poly_degradation"]) < abs(r["seq_degradation"]))

    print(f"\n=== Summary ===")
    print(f"  Mean single-skip degradation: poly={poly_mean_single:.4f}  "
          f"seq={seq_mean_single:.4f}")
    print(f"  Poly more robust on {poly_wins}/{n_layers} layers")

    result = {
        "poly_checkpoint": args.poly_ckpt,
        "seq_checkpoint": args.seq_ckpt,
        "n_layers": n_layers,
        "poly_baseline_bpb": float(poly_base),
        "seq_baseline_bpb": float(seq_base),
        "single_layer_skip": single_results,
        "multi_layer_skip": multi_results,
        "summary": {
            "poly_mean_single_degradation": float(poly_mean_single),
            "seq_mean_single_degradation": float(seq_mean_single),
            "poly_wins_single": poly_wins,
            "poly_wins_fraction": poly_wins / n_layers,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
