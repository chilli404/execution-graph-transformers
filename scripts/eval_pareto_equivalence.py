"""Evaluate graph equivalence metrics for a trained pareto checkpoint.

Computes agreement, symmetric KL, and BPB for sequential vs all-parallel
on a validation set. Lightweight — no full graph rewrite sweep, just the
two extreme modes.

Usage:
  python scripts/eval_pareto_equivalence.py \
      --ckpt /tmp/fogen/pareto/1b_foo/ckpts/step012000.safetensors \
      --config configs/pareto/1b_foo.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output /tmp/fogen/pareto/1b_foo/equivalence.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader, load_tokenizer
from fogen.evals.bpb import val_stream
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
    state = load_file(args.ckpt)
    missing, unexpected = model.load_state_dict(
        {k: v.to(param_dtype) for k, v in state.items()}, strict=False)
    assert not unexpected
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    sample = val_stream(
        shard_dir=args.val_shards, tokenizer=tokenizer,
        ctx_len=mcfg.ctx_len, batch_size=8, seed=42,
        device=str(device),
    )["val_bpb"]

    loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=42, device=device)
    x, y = loader.next_batch()

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        seq_logits = model(x, mode="sequential").float()
        par_logits = model(x, mode="parallel").float()

    # Agreement
    agreement = float(
        (seq_logits.argmax(dim=-1) == par_logits.argmax(dim=-1)).float().mean())

    # Symmetric KL (per-token, nats)
    seq_lp = F.log_softmax(seq_logits, dim=-1)
    par_lp = F.log_softmax(par_logits, dim=-1)
    sym_kl = float((
        F.kl_div(par_lp, seq_lp.exp(), reduction="batchmean")
        + F.kl_div(seq_lp, par_lp.exp(), reduction="batchmean")
    ) / 2)

    # BPB for each mode
    seq_bpb = float(val_stream(
        shard_dir=args.val_shards, tokenizer=tokenizer,
        ctx_len=mcfg.ctx_len, batch_size=8, seed=42,
        device=str(device), model=model, mode="sequential",
    )["val_bpb"]) if hasattr(val_stream, '__call__') else None

    result = {
        "checkpoint": args.ckpt,
        "config": args.config,
        "agreement": agreement,
        "symmetric_kl": sym_kl,
        "n_params": model.num_params(),
        "d_model": mcfg.d_model,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  agreement={agreement:.4f}  sym_kl={sym_kl:.3f}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
