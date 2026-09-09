"""Measure layer-skip robustness for a single model across all layers.

Reports BPB degradation when each layer is skipped, and multi-layer
skip degradation curves. No specialist comparison — just measures
how gracefully the model handles layer removal.

Usage:
  python scripts/eval_layer_skip_single.py \
      --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/layer_skip_430m_poly.json
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

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, val_stream
from fogen.model import GPT, ModelConfig


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

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    token_bytes = torch.tensor(
        [len(tokenizer.decode([i]).encode("utf-8")) for i in range(mcfg.vocab_size)],
        dtype=torch.long)

    n_layers = mcfg.n_layer

    def eval_bpb(mask):
        old = model.cfg.execution_mode
        model.cfg.execution_mode = mask
        result = evaluate_bpb(model, stream, token_bytes, mcfg.ctx_len,
                              batch_size=8, max_windows=64, device=str(device))
        model.cfg.execution_mode = old
        return result["val_bpb"]

    # Baseline
    print(f"Model: {args.ckpt}", flush=True)
    print(f"Layers: {n_layers}", flush=True)
    baseline = eval_bpb("sequential")
    par_baseline = eval_bpb("parallel")
    print(f"Seq baseline BPB: {baseline:.4f}")
    print(f"Par baseline BPB: {par_baseline:.4f}")

    # Single-layer skip
    print(f"\n=== Single-layer skip ===", flush=True)
    single = []
    for i in range(n_layers):
        mask = ["sequential"] * n_layers
        mask[i] = "skip"
        bpb = eval_bpb(mask)
        deg = bpb - baseline
        single.append({"layer": i, "bpb": float(bpb), "degradation": float(deg)})
        print(f"  L{i:>2}: Δ={deg:+.4f}", flush=True)

    # Multi-layer skip (ordered by least impactful first)
    order = sorted(range(n_layers), key=lambda i: single[i]["degradation"])
    print(f"\n=== Multi-layer skip (best-first order) ===", flush=True)
    multi = []
    for n_skip in range(1, min(n_layers, 11)):
        mask = ["sequential"] * n_layers
        for i in order[:n_skip]:
            mask[i] = "skip"
        bpb = eval_bpb(mask)
        deg = bpb - baseline
        multi.append({
            "n_skip": n_skip,
            "skipped_layers": sorted(order[:n_skip]),
            "bpb": float(bpb),
            "degradation": float(deg),
        })
        print(f"  Skip {n_skip:>2}: Δ={deg:+.4f} (layers {sorted(order[:n_skip])})",
              flush=True)

    result = {
        "checkpoint": args.ckpt,
        "config": args.config,
        "n_layers": n_layers,
        "n_params": model.num_params(),
        "seq_baseline_bpb": float(baseline),
        "par_baseline_bpb": float(par_baseline),
        "mean_single_degradation": float(np.mean([s["degradation"] for s in single])),
        "single_layer_skip": single,
        "multi_layer_skip": multi,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
