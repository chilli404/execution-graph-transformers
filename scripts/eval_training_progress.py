"""Measure agreement, KL, and β at intermediate training checkpoints.

Checks whether β changes during training and whether the cross-scale
β difference is confounded with training stage.

Usage:
  python scripts/eval_training_progress.py \
      --ckpt_dir runs/430m_poly_full/ckpts \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/430m_training_progress.json \
      [--n_masks 200]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from scipy.stats import spearmanr

from fogen.evals.bpb import val_stream
from fogen.execution_graph import fit_count_adjusted_scale
from fogen.model import GPT, ModelConfig


def load_model(checkpoint, model_cfg, device):
    model = GPT(model_cfg).to(device)
    state = {k: v.float() for k, v in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(k.startswith("rope_") for k in missing)
    return model.eval()


def symmetric_kl(logits_a, logits_b):
    """Per-token symmetric KL in nats (batchmean = sum over vocab, mean over positions)."""
    lp_a = F.log_softmax(logits_a, dim=-1)
    lp_b = F.log_softmax(logits_b, dim=-1)
    return float((
        F.kl_div(lp_a, lp_b.exp(), reduction="batchmean")
        + F.kl_div(lp_b, lp_a.exp(), reduction="batchmean")
    ) / 2)


def evaluate_checkpoint(model, sample, n_layers, n_masks, rng, device):
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        seq_logits = model(sample, mode="sequential").float()
        par_logits = model(sample, mode="parallel").float()
        layer_defects = model.layer_defects(sample).float().cpu().numpy()

    agreement = float(
        (seq_logits.argmax(-1) == par_logits.argmax(-1)).float().mean())
    all_par_kl = symmetric_kl(seq_logits, par_logits)

    seq_logprob = F.log_softmax(seq_logits, dim=-1)
    seq_prob = seq_logprob.exp()

    # Single-layer KL effects
    single_kl = np.zeros(n_layers)
    for layer in range(n_layers):
        mask = ["parallel" if i == layer else "sequential" for i in range(n_layers)]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(sample, mode=mask).float()
        lp = F.log_softmax(logits, dim=-1)
        single_kl[layer] = float((
            F.kl_div(lp, seq_prob, reduction="batchmean")
            + F.kl_div(seq_logprob, lp.exp(), reduction="batchmean")
        ) / 2)

    # Random multi-layer masks for β fitting
    pred_sums, actual_kls, n_pars = [], [], []
    for _ in range(n_masks):
        bits = rng.integers(0, 2, size=n_layers)
        n_par = int(bits.sum())
        if n_par < 2:
            continue
        mask = ["parallel" if b else "sequential" for b in bits]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(sample, mode=mask).float()
        lp = F.log_softmax(logits, dim=-1)
        kl = float((
            F.kl_div(lp, seq_prob, reduction="batchmean")
            + F.kl_div(seq_logprob, lp.exp(), reduction="batchmean")
        ) / 2)
        pred_sums.append(float(bits @ single_kl))
        actual_kls.append(kl)
        n_pars.append(n_par)

    alpha0, beta = None, None
    if len(pred_sums) >= 10:
        alpha0, beta = fit_count_adjusted_scale(pred_sums, actual_kls, n_pars)

    return {
        "agreement": agreement,
        "all_par_kl": all_par_kl,
        "layer_defects": layer_defects.tolist(),
        "mean_defect": float(layer_defects.mean()),
        "single_layer_kl": single_kl.tolist(),
        "alpha0": alpha0,
        "beta": beta,
        "n_masks_used": len(pred_sums),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_masks", type=int, default=200)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_cfg = ModelConfig(**cfg["model"])
    n_layers = model_cfg.n_layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stream = val_stream(args.val_shards)
    context = cfg["model"]["ctx_len"]
    sample = torch.tensor(
        np.asarray(stream[:4 * context], dtype=np.int64).reshape(4, context),
        device=device,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("step*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    rng = np.random.default_rng(2026)
    results = []
    for ckpt_path in ckpts:
        step = int(re.search(r"step(\d+)", ckpt_path.name).group(1))
        print(f"\n=== Step {step}: {ckpt_path.name} ===", flush=True)
        model = load_model(str(ckpt_path), model_cfg, device)
        ckpt_rng = np.random.default_rng(rng.integers(0, 2**32))
        record = evaluate_checkpoint(
            model, sample, n_layers, args.n_masks, ckpt_rng, device)
        record["step"] = step
        record["checkpoint"] = str(ckpt_path)
        results.append(record)
        print(f"  agreement={record['agreement']:.4f}  "
              f"all_par_kl={record['all_par_kl']:.4f}  "
              f"beta={record['beta']}", flush=True)
        del model
        torch.cuda.empty_cache()

    output = {
        "config": args.config,
        "n_layers": n_layers,
        "n_masks": args.n_masks,
        "checkpoints": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary
    print(f"\n{'Step':>8} {'Agreement':>10} {'All-par KL':>11} {'β':>8} {'Mean defect':>12}")
    print("-" * 55)
    for r in results:
        b = f"{r['beta']:.4f}" if r['beta'] is not None else "N/A"
        print(f"{r['step']:>8} {r['agreement']:>10.4f} {r['all_par_kl']:>11.4f} "
              f"{b:>8} {r['mean_defect']:>12.6f}")


if __name__ == "__main__":
    main()
