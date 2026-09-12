"""Cross-sequence-length composition law transfer test.

Fits the composition model (α₀, β) on defects measured at one sequence
length and evaluates predictions on a different sequence length. If the
law transfers, it describes the model's weight structure rather than
the evaluation context.

Usage:
    PYTHONPATH=src python scripts/eval_composition_cross_seqlen.py \
        --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
        --config blackwell/configs/scale430m_poly_full.yaml \
        --val_shards data/climbmix/bpe8192/shards \
        --output blackwell/results/430m_cross_seqlen.json \
        --fit_seqlen 128 --test_seqlens 256 512 1024 \
        --n_masks 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

from fogen.evals.bpb import val_stream
from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def measure_at_seqlen(model, stream, seqlen, masks, device):
    """Measure layer defects and graph KLs at a specific sequence length."""
    n_ctx = min(seqlen, model.cfg.ctx_len)
    n_seqs = max(1, 2 * model.cfg.ctx_len // n_ctx)
    total_tokens = n_seqs * n_ctx
    tokens_flat = np.asarray(stream[:total_tokens], dtype=np.int64)
    sample = torch.tensor(
        tokens_flat[:n_seqs * n_ctx].reshape(n_seqs, n_ctx), device=device)

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        sequential_logits = model(sample, mode="sequential").float()
        layer_defects = model.layer_defects(sample).float().cpu().numpy()

    sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
    sequential_probability = sequential_logprob.exp()

    # Measure single-layer effects
    n_layers = len(model.blocks)
    single_kls = np.zeros(n_layers)
    for layer in range(n_layers):
        mask = ["parallel" if i == layer else "sequential" for i in range(n_layers)]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(sample, mode=mask).float()
        logprob = F.log_softmax(logits, dim=-1)
        kl = float((
            F.kl_div(logprob, sequential_probability, reduction="batchmean")
            + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
        ) / 2)
        single_kls[layer] = kl

    baseline_kl = 0.0  # all-sequential vs all-sequential = 0

    # Measure multi-layer graph KLs
    results = []
    for bits in masks:
        if sum(bits) < 2:
            continue
        mask = ["parallel" if b else "sequential" for b in bits]
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(sample, mode=mask).float()
        logprob = F.log_softmax(logits, dim=-1)
        kl = float((
            F.kl_div(logprob, sequential_probability, reduction="batchmean")
            + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
        ) / 2)
        pred_sum = sum(single_kls[i] for i, b in enumerate(bits) if b)
        results.append({
            "bits": list(bits),
            "n_parallel": sum(bits),
            "predicted_sum": float(pred_sum),
            "actual_kl": float(kl),
        })

    return {
        "seqlen": n_ctx,
        "layer_defects": layer_defects.tolist(),
        "single_kls": single_kls.tolist(),
        "graphs": results,
    }


def fit_2p(X, y, n):
    def loss(p):
        return float(np.mean((y - p[0] * X * np.power(n, -p[1]))**2))
    best = None
    for a0 in [0.5, 1.0, 1.3, 2.0]:
        for b0 in [0.1, 0.27, 0.4]:
            res = minimize(loss, [a0, b0], method='Nelder-Mead',
                          options={'xatol': 1e-9, 'fatol': 1e-13, 'maxiter': 10000})
            if best is None or res.fun < best.fun:
                best = res
    return best.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit_seqlen", type=int, default=128)
    parser.add_argument("--test_seqlens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--n_masks", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, args.config, device)
    stream = val_stream(args.val_shards)
    n_layers = len(model.blocks)

    # Generate shared random masks
    rng = np.random.default_rng(2026)
    masks = []
    mask_set = set()
    # Include all single-layer masks
    for i in range(n_layers):
        m = tuple(int(j == i) for j in range(n_layers))
        masks.append(m)
        mask_set.add(m)
    # Random masks
    while len(masks) < args.n_masks:
        m = tuple(rng.integers(0, 2, size=n_layers).tolist())
        if m not in mask_set and sum(m) >= 2:
            masks.append(m)
            mask_set.add(m)

    all_seqlens = [args.fit_seqlen] + [s for s in args.test_seqlens if s != args.fit_seqlen]

    measurements = {}
    for seqlen in all_seqlens:
        print(f"\nMeasuring at seqlen={seqlen}...", flush=True)
        measurements[seqlen] = measure_at_seqlen(model, stream, seqlen, masks, device)
        print(f"  {len(measurements[seqlen]['graphs'])} multi-layer graphs measured")

    # Fit on fit_seqlen
    fit_data = measurements[args.fit_seqlen]
    X_fit = np.array([g["predicted_sum"] for g in fit_data["graphs"]])
    y_fit = np.array([g["actual_kl"] for g in fit_data["graphs"]])
    n_fit = np.array([g["n_parallel"] for g in fit_data["graphs"]], dtype=float)

    params = fit_2p(X_fit, y_fit, n_fit)
    alpha_1p = float(np.dot(X_fit, y_fit) / np.dot(X_fit, X_fit))

    print(f"\nFit on seqlen={args.fit_seqlen}:")
    print(f"  1p: α = {alpha_1p:.4f}")
    print(f"  2p: α₀ = {params[0]:.4f}, β = {params[1]:.4f}")

    # Evaluate on all sequence lengths (including fit length for reference)
    output = {
        "fit_seqlen": args.fit_seqlen,
        "fit_alpha_1p": alpha_1p,
        "fit_alpha0": float(params[0]),
        "fit_beta": float(params[1]),
        "n_masks": len(masks),
        "evaluations": {},
    }

    print(f"\n{'SeqLen':>7} | {'1p Pearson':>10} | {'2p Pearson':>10} | {'1p RMSE':>8} | {'2p RMSE':>8} | {'Spearman':>8}")
    print(f"{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for seqlen, data in measurements.items():
        X = np.array([g["predicted_sum"] for g in data["graphs"]])
        y = np.array([g["actual_kl"] for g in data["graphs"]])
        n = np.array([g["n_parallel"] for g in data["graphs"]], dtype=float)

        # Predict using fit_seqlen parameters
        pred_1p = alpha_1p * X
        pred_2p = params[0] * X * np.power(n, -params[1])

        rmse_1p = float(np.sqrt(np.mean((y - pred_1p)**2)) / np.sqrt(np.mean(y**2)))
        rmse_2p = float(np.sqrt(np.mean((y - pred_2p)**2)) / np.sqrt(np.mean(y**2)))
        r_1p = float(pearsonr(pred_1p, y).statistic)
        r_2p = float(pearsonr(pred_2p, y).statistic)
        rho = float(spearmanr(pred_2p, y).statistic)

        is_fit = " (fit)" if seqlen == args.fit_seqlen else ""
        print(f"{seqlen:>7} | {r_1p:>10.4f} | {r_2p:>10.4f} | {rmse_1p:>8.4f} | {rmse_2p:>8.4f} | {rho:>8.4f}{is_fit}")

        # Also fit locally for comparison
        local_params = fit_2p(X, y, n)

        output["evaluations"][str(seqlen)] = {
            "n_graphs": len(data["graphs"]),
            "transfer_1p_pearson": r_1p,
            "transfer_2p_pearson": r_2p,
            "transfer_1p_rmse": rmse_1p,
            "transfer_2p_rmse": rmse_2p,
            "transfer_spearman": rho,
            "local_alpha0": float(local_params[0]),
            "local_beta": float(local_params[1]),
        }

    # Defect cross-correlation
    print(f"\n--- Layer defect correlation across sequence lengths ---")
    fit_defects = np.array(measurements[args.fit_seqlen]["layer_defects"])
    for seqlen, data in measurements.items():
        if seqlen == args.fit_seqlen:
            continue
        test_defects = np.array(data["layer_defects"])
        rho = float(spearmanr(fit_defects, test_defects).statistic)
        print(f"  seqlen {args.fit_seqlen} vs {seqlen}: Spearman = {rho:.4f}")
        output["evaluations"][str(seqlen)]["defect_cross_spearman"] = rho

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
