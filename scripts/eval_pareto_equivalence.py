"""Evaluate graph equivalence metrics for a trained pareto checkpoint.

Computes agreement and symmetric KL between sequential and all-parallel
on a validation batch. Lightweight — no full graph rewrite sweep.

Usage:
  python scripts/eval_pareto_equivalence.py \
      --ckpt /tmp/ckpt.safetensors \
      --config configs/pareto/1b_foo.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --bf16 \
      --output results/equivalence.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader
from fogen.model import GPT, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--n_batches", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = GPT(mcfg).to(device=device, dtype=param_dtype)
    state = load_file(args.ckpt)
    missing, unexpected = model.load_state_dict(
        {k: v.to(param_dtype) for k, v in state.items()}, strict=False)
    assert not unexpected
    model.eval()

    loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=42, device=device)

    agreements = []
    kls = []

    for _ in range(args.n_batches):
        x, y = loader.next_batch()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            seq_logits = model(x, mode="sequential").float()
            par_logits = model(x, mode="parallel").float()

        agreement = float(
            (seq_logits.argmax(dim=-1) == par_logits.argmax(dim=-1)).float().mean())
        agreements.append(agreement)

        seq_lp = F.log_softmax(seq_logits, dim=-1)
        par_lp = F.log_softmax(par_logits, dim=-1)
        sym_kl = float((
            F.kl_div(par_lp, seq_lp.exp(), reduction="batchmean")
            + F.kl_div(seq_lp, par_lp.exp(), reduction="batchmean")
        ) / 2)
        kls.append(sym_kl)

    import numpy as np
    result = {
        "checkpoint": args.ckpt,
        "config": args.config,
        "agreement": float(np.mean(agreements)),
        "agreement_std": float(np.std(agreements)),
        "symmetric_kl": float(np.mean(kls)),
        "symmetric_kl_std": float(np.std(kls)),
        "n_batches": args.n_batches,
        "n_params": model.num_params(),
        "d_model": mcfg.d_model,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  agreement={result['agreement']:.4f}  sym_kl={result['symmetric_kl']:.3f}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
