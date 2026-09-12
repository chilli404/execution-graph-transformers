"""Autoregressive generation divergence test between execution modes.

Generates greedily from the same prompts under sequential and parallel
modes, then measures how quickly and how often the outputs diverge.

Usage:
  python scripts/eval_generation_divergence.py \
      --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/generation_divergence.json
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
from fogen.evals.bpb import val_stream
from fogen.model import GPT, ModelConfig


@torch.no_grad()
def greedy_generate(model, prompt_ids, gen_length, mode, device):
    """Generate greedily using KV-cache, one token at a time."""
    idx = prompt_ids.unsqueeze(0).to(device)  # (1, prompt_len)
    cache = None

    # Prefill: run the full prompt
    logits, cache = model.forward_cached(idx, cache=cache, mode=mode)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
    generated = [next_token.item()]

    # Decode: one token at a time
    for _ in range(gen_length - 1):
        logits, cache = model.forward_cached(next_token, cache=cache, mode=mode)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_prompts", type=int, default=200)
    parser.add_argument("--prompt_length", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=256)
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

    # Extract prompts from validation stream
    stream = val_stream(args.val_shards)
    stride = args.prompt_length + args.gen_length
    n_available = (len(stream) - 1) // stride
    n_prompts = min(args.n_prompts, n_available)

    print(f"Model: {args.ckpt}")
    print(f"Prompts: {n_prompts} × {args.prompt_length} tokens → generate {args.gen_length}")
    print(f"Comparing: sequential vs parallel", flush=True)

    modes = ["sequential", "parallel"]
    identical_count = 0
    first_divergence = []
    total_agree = 0
    total_tokens = 0
    per_prompt = []

    for i in range(n_prompts):
        start = i * stride
        prompt = torch.tensor(stream[start:start + args.prompt_length], dtype=torch.long)

        generations = {}
        for mode in modes:
            generations[mode] = greedy_generate(
                model, prompt, args.gen_length, mode, device)

        seq_gen = generations["sequential"]
        par_gen = generations["parallel"]

        # Compare
        identical = (seq_gen == par_gen)
        if identical:
            identical_count += 1
            first_div = args.gen_length  # no divergence
        else:
            first_div = next(
                (j for j in range(args.gen_length) if seq_gen[j] != par_gen[j]),
                args.gen_length)
            first_divergence.append(first_div)

        agree = sum(1 for a, b in zip(seq_gen, par_gen) if a == b)
        total_agree += agree
        total_tokens += args.gen_length

        per_prompt.append({
            "prompt_idx": i,
            "identical": identical,
            "first_divergence": first_div,
            "token_agreement": agree / args.gen_length,
        })

        if (i + 1) % 20 == 0 or i == 0:
            pct = identical_count / (i + 1) * 100
            print(f"  [{i+1}/{n_prompts}] {pct:.0f}% identical so far", flush=True)

    # Summary
    frac_identical = identical_count / n_prompts
    token_agreement = total_agree / total_tokens
    mean_first_div = np.mean(first_divergence) if first_divergence else args.gen_length
    median_first_div = np.median(first_divergence) if first_divergence else args.gen_length

    print(f"\n=== Results ===")
    print(f"  Identical sequences: {identical_count}/{n_prompts} ({frac_identical:.1%})")
    print(f"  Token agreement: {token_agreement:.4f}")
    if first_divergence:
        print(f"  First divergence (non-identical): mean={mean_first_div:.1f}, "
              f"median={median_first_div:.1f}")
        # Distribution
        buckets = [0, 10, 25, 50, 100, 150, 200, 256]
        print(f"  Divergence distribution:")
        for lo, hi in zip(buckets, buckets[1:]):
            count = sum(1 for d in first_divergence if lo <= d < hi)
            if count > 0:
                print(f"    tokens {lo}-{hi}: {count} sequences")

    result = {
        "checkpoint": args.ckpt,
        "n_prompts": n_prompts,
        "prompt_length": args.prompt_length,
        "gen_length": args.gen_length,
        "fraction_identical": frac_identical,
        "token_agreement": token_agreement,
        "mean_first_divergence": float(mean_first_div),
        "median_first_divergence": float(median_first_div),
        "n_diverged": len(first_divergence),
        "first_divergence_values": first_divergence,
        "per_prompt": per_prompt,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
