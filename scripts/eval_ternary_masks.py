"""Evaluate held-out ternary masks on ternary-trained vs binary-only models.

Samples thousands of random {sequential, parallel, skip}^L masks and
measures BPB degradation for each, comparing a ternary-trained model
against a binary-only (seq/par) polymorphic model.

Usage:
  python scripts/eval_ternary_masks.py \
      --ternary_ckpt runs/120m_ternary_poly/ckpts/step003500.safetensors \
      --binary_ckpt runs/120m_poly_cw01/ckpts/step003500.safetensors \
      --ternary_config configs/scale120m_ternary_polymorphic.yaml \
      --binary_config configs/scale120m_polymorphic_cw01.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --n_masks 5000 \
      --output blackwell/results/ternary_mask_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, val_stream
from fogen.model import GPT, ModelConfig

MODE_NAMES = ["sequential", "parallel", "skip"]


def load_model(ckpt, config_path, device):
    cfg = yaml.safe_load(open(config_path))
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(ckpt).items()}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("rope_") for k in missing)
    return model.eval(), mcfg


def eval_bpb(model, stream, token_bytes, ctx_len, mask, device, max_windows=32):
    old = model.cfg.execution_mode
    model.cfg.execution_mode = mask
    result = evaluate_bpb(model, stream, token_bytes, ctx_len,
                          batch_size=8, max_windows=max_windows,
                          device=str(device))
    model.cfg.execution_mode = old
    return result["val_bpb"]


def sample_ternary_masks(n_layers, n_masks, rng, probs=(0.4, 0.4, 0.2)):
    """Sample random ternary masks with given mode probabilities."""
    masks = []
    cumprobs = [probs[0], probs[0] + probs[1]]
    for _ in range(n_masks):
        r = rng.random(n_layers)
        mask = []
        for v in r:
            if v < cumprobs[0]:
                mask.append("sequential")
            elif v < cumprobs[1]:
                mask.append("parallel")
            else:
                mask.append("skip")
        masks.append(mask)
    return masks


def mask_stats(mask):
    return {
        "n_seq": mask.count("sequential"),
        "n_par": mask.count("parallel"),
        "n_skip": mask.count("skip"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ternary_ckpt", required=True)
    parser.add_argument("--binary_ckpt", required=True)
    parser.add_argument("--ternary_config", required=True)
    parser.add_argument("--binary_config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--n_masks", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading ternary model...", flush=True)
    ternary_model, mcfg = load_model(args.ternary_ckpt, args.ternary_config, device)
    print("Loading binary model...", flush=True)
    binary_model, _ = load_model(args.binary_ckpt, args.binary_config, device)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    token_bytes = torch.tensor(
        [len(tokenizer.decode([i]).encode("utf-8")) for i in range(mcfg.vocab_size)],
        dtype=torch.long)

    n_layers = mcfg.n_layer

    # Baselines
    print("\n=== Baselines ===", flush=True)
    tern_seq = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                        "sequential", device, max_windows=64)
    tern_par = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                        "parallel", device, max_windows=64)
    bin_seq = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                       "sequential", device, max_windows=64)
    bin_par = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                       "parallel", device, max_windows=64)
    print(f"  Ternary: seq={tern_seq:.4f}  par={tern_par:.4f}")
    print(f"  Binary:  seq={bin_seq:.4f}  par={bin_par:.4f}")

    # Sample masks matching training distribution (40% seq, 40% par, 20% skip)
    rng = np.random.default_rng(42)
    masks = sample_ternary_masks(n_layers, args.n_masks, rng, probs=(0.4, 0.4, 0.2))

    print(f"\n=== Evaluating {args.n_masks} ternary masks ===", flush=True)
    rows = []
    for i, mask in enumerate(masks):
        stats = mask_stats(mask)
        tern_bpb = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                            mask, device)
        bin_bpb = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                           mask, device)
        rows.append({
            "mask": mask,
            **stats,
            "ternary_bpb": float(tern_bpb),
            "binary_bpb": float(bin_bpb),
            "ternary_degradation": float(tern_bpb - tern_seq),
            "binary_degradation": float(bin_bpb - bin_seq),
        })
        if (i + 1) % 100 == 0:
            tern_degs = [r["ternary_degradation"] for r in rows]
            bin_degs = [r["binary_degradation"] for r in rows]
            print(f"  [{i+1}/{args.n_masks}] tern mean Δ={np.mean(tern_degs):.4f}  "
                  f"bin mean Δ={np.mean(bin_degs):.4f}", flush=True)

    # Summary statistics
    tern_degs = np.array([r["ternary_degradation"] for r in rows])
    bin_degs = np.array([r["binary_degradation"] for r in rows])

    print(f"\n=== Summary ===")
    print(f"{'':>25} {'Ternary':>10} {'Binary':>10}")
    print(f"{'Mean Δ':>25} {np.mean(tern_degs):>10.4f} {np.mean(bin_degs):>10.4f}")
    print(f"{'Median Δ':>25} {np.median(tern_degs):>10.4f} {np.median(bin_degs):>10.4f}")
    print(f"{'95th pct Δ':>25} {np.percentile(tern_degs, 95):>10.4f} {np.percentile(bin_degs, 95):>10.4f}")
    print(f"{'Max Δ':>25} {np.max(tern_degs):>10.4f} {np.max(bin_degs):>10.4f}")
    print(f"{'Frac Δ < 0.01':>25} {np.mean(tern_degs < 0.01):>10.3f} {np.mean(bin_degs < 0.01):>10.3f}")
    print(f"{'Frac Δ < 0.05':>25} {np.mean(tern_degs < 0.05):>10.3f} {np.mean(bin_degs < 0.05):>10.3f}")
    print(f"{'Frac Δ < 0.10':>25} {np.mean(tern_degs < 0.10):>10.3f} {np.mean(bin_degs < 0.10):>10.3f}")

    # Breakdown by skip count
    print(f"\n=== By skip count ===")
    print(f"{'n_skip':>8} {'n':>6} {'tern mean Δ':>12} {'bin mean Δ':>12}")
    for ns in range(n_layers + 1):
        idx = [i for i, r in enumerate(rows) if r["n_skip"] == ns]
        if len(idx) < 5:
            continue
        t = np.mean(tern_degs[idx])
        b = np.mean(bin_degs[idx])
        print(f"{ns:>8} {len(idx):>6} {t:>12.4f} {b:>12.4f}")

    result = {
        "ternary_checkpoint": args.ternary_ckpt,
        "binary_checkpoint": args.binary_ckpt,
        "n_layers": n_layers,
        "n_masks": len(rows),
        "baselines": {
            "ternary_seq": float(tern_seq),
            "ternary_par": float(tern_par),
            "binary_seq": float(bin_seq),
            "binary_par": float(bin_par),
        },
        "summary": {
            "ternary_mean_degradation": float(np.mean(tern_degs)),
            "ternary_median_degradation": float(np.median(tern_degs)),
            "ternary_p95_degradation": float(np.percentile(tern_degs, 95)),
            "ternary_max_degradation": float(np.max(tern_degs)),
            "binary_mean_degradation": float(np.mean(bin_degs)),
            "binary_median_degradation": float(np.median(bin_degs)),
            "binary_p95_degradation": float(np.percentile(bin_degs, 95)),
            "binary_max_degradation": float(np.max(bin_degs)),
            "ternary_frac_lt_001": float(np.mean(tern_degs < 0.01)),
            "ternary_frac_lt_005": float(np.mean(tern_degs < 0.05)),
            "ternary_frac_lt_010": float(np.mean(tern_degs < 0.10)),
            "binary_frac_lt_001": float(np.mean(bin_degs < 0.01)),
            "binary_frac_lt_005": float(np.mean(bin_degs < 0.05)),
            "binary_frac_lt_010": float(np.mean(bin_degs < 0.10)),
        },
        "rows": rows,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
