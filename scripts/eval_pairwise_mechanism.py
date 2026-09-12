"""Full C(L,2) pairwise interaction analysis + defect vector cosine geometry.

Evaluates ALL layer pairs to compute interactions and tests whether
defect-vector alignment predicts cancellation — the mechanistic link
between local Jacobian geometry and the empirical composition law.

Usage:
  python scripts/eval_pairwise_mechanism.py \
      --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/430m_pairwise_mechanism.json
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

from fogen.evals.bpb import val_stream
from fogen.model import GPT, ModelConfig, _rmsnorm


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {k: v.float() for k, v in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(k.startswith("rope_") for k in missing)
    return model.eval(), cfg


def compute_symmetric_kl(model, sample, mask, seq_logprob, seq_prob, device):
    """Per-token symmetric KL in nats for a given execution mask."""
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model(sample, mode=mask).float()
    lp = F.log_softmax(logits, dim=-1)
    return float((
        F.kl_div(lp, seq_prob, reduction="batchmean")
        + F.kl_div(seq_logprob, lp.exp(), reduction="batchmean")
    ) / 2)


def extract_defect_vectors(model, sample, device):
    """Extract per-layer defect vectors: d_l = sequential_output - parallel_output.

    Returns a list of tensors, each of shape (B, T, D), one per layer.
    The defect vector at layer l captures how switching that layer from
    sequential to parallel changes the hidden state at that layer's output.
    """
    n_layers = len(model.blocks)
    T = sample.size(1)
    cos = model.rope_cos[:, :, :T]
    sin = model.rope_sin[:, :, :T]

    defect_vectors = []
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        x = _rmsnorm(model.wte(sample))
        for layer_idx, block in enumerate(model.blocks):
            ve_key = str(layer_idx)
            ve = model.value_embeds[ve_key](sample) if ve_key in model.value_embeds else None
            normalized = _rmsnorm(x)
            attention = block.attn(normalized, ve, cos, sin)

            # Sequential: attn -> norm -> ffn
            attended = x + attention
            sequential_out = attended + block.mlp(_rmsnorm(attended))

            # Parallel: attn and ffn from same normalized input
            parallel_out = x + attention + block.mlp(normalized)

            defect = (sequential_out - parallel_out).float()
            defect_vectors.append(defect)

            # Continue with sequential path for next layer
            x = sequential_out

    return defect_vectors


def pairwise_cosine(vec_i, vec_j):
    """Mean cosine similarity between two defect vector fields.

    vec_i, vec_j: (B, T, D). Computes per-position cosine, then averages.
    """
    # Flatten B,T into one dimension
    flat_i = vec_i.reshape(-1, vec_i.size(-1))
    flat_j = vec_j.reshape(-1, vec_j.size(-1))
    norm_i = flat_i.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    norm_j = flat_j.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_sim = (flat_i / norm_i * flat_j / norm_j).sum(dim=-1)
    return float(cos_sim.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, args.config, device)
    n_layers = cfg["model"]["n_layer"]
    context = cfg["model"]["ctx_len"]

    stream = val_stream(args.val_shards)
    sample = torch.tensor(
        np.asarray(stream[:4 * context], dtype=np.int64).reshape(4, context),
        device=device,
    )

    # 1. Sequential baseline logits
    print("Computing sequential baseline...", flush=True)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        seq_logits = model(sample, mode="sequential").float()
    seq_logprob = F.log_softmax(seq_logits, dim=-1)
    seq_prob = seq_logprob.exp()

    # 2. Single-layer KL
    print(f"Computing {n_layers} single-layer KL values...", flush=True)
    single_kl = {}
    for layer in range(n_layers):
        mask = ["parallel" if i == layer else "sequential" for i in range(n_layers)]
        kl = compute_symmetric_kl(model, sample, mask, seq_logprob, seq_prob, device)
        single_kl[layer] = kl
        print(f"  L{layer}: KL={kl:.4f}", flush=True)

    # 3. All C(L,2) pair KL values
    pairs = list(itertools.combinations(range(n_layers), 2))
    print(f"Computing {len(pairs)} pair KL values...", flush=True)
    pair_kl = {}
    pair_interaction = {}
    for idx, (i, j) in enumerate(pairs):
        mask = ["parallel" if l in (i, j) else "sequential" for l in range(n_layers)]
        kl = compute_symmetric_kl(model, sample, mask, seq_logprob, seq_prob, device)
        interaction = kl - single_kl[i] - single_kl[j]
        pair_kl[(i, j)] = kl
        pair_interaction[(i, j)] = interaction
        if (idx + 1) % 20 == 0 or idx == len(pairs) - 1:
            print(f"  [{idx+1}/{len(pairs)}] ({i},{j}): KL={kl:.4f}, "
                  f"I={interaction:+.4f}", flush=True)

    # 4. Extract defect vectors and compute pairwise cosines
    print("Extracting defect vectors...", flush=True)
    defect_vectors = extract_defect_vectors(model, sample, device)

    print(f"Computing {len(pairs)} pairwise cosine similarities...", flush=True)
    pair_cosine = {}
    for i, j in pairs:
        cos = pairwise_cosine(defect_vectors[i], defect_vectors[j])
        pair_cosine[(i, j)] = cos

    # Free memory
    del defect_vectors
    torch.cuda.empty_cache()

    # 5. Correlations
    interactions = [pair_interaction[(i, j)] for i, j in pairs]
    cosines = [pair_cosine[(i, j)] for i, j in pairs]
    distances = [abs(i - j) for i, j in pairs]

    cos_vs_int_spearman = spearmanr(cosines, interactions)
    cos_vs_int_pearson = pearsonr(cosines, interactions)
    dist_vs_int_spearman = spearmanr(distances, interactions)
    dist_vs_cos_spearman = spearmanr(distances, cosines)

    # 6. Summary statistics
    n_subadditive = sum(1 for v in interactions if v < 0)
    mean_interaction = float(np.mean(interactions))
    mean_cosine = float(np.mean(cosines))

    print(f"\n{'=' * 60}")
    print(f"Pairwise Mechanism Summary ({n_layers} layers, {len(pairs)} pairs)")
    print(f"{'=' * 60}")
    print(f"Subadditive pairs: {n_subadditive}/{len(pairs)} "
          f"({100*n_subadditive/len(pairs):.1f}%)")
    print(f"Mean interaction: {mean_interaction:.4f}")
    print(f"Mean defect cosine: {mean_cosine:.4f}")
    print(f"\ncos(d_i,d_j) vs I_ij:")
    print(f"  Spearman: {cos_vs_int_spearman.statistic:.4f} "
          f"(p={cos_vs_int_spearman.pvalue:.4e})")
    print(f"  Pearson:  {cos_vs_int_pearson.statistic:.4f} "
          f"(p={cos_vs_int_pearson.pvalue:.4e})")
    print(f"\n|i-j| vs I_ij:")
    print(f"  Spearman: {dist_vs_int_spearman.statistic:.4f} "
          f"(p={dist_vs_int_spearman.pvalue:.4e})")
    print(f"\n|i-j| vs cos(d_i,d_j):")
    print(f"  Spearman: {dist_vs_cos_spearman.statistic:.4f} "
          f"(p={dist_vs_cos_spearman.pvalue:.4e})")

    # Build full matrices for JSON output
    interaction_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    cosine_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    kl_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    for (i, j), val in pair_interaction.items():
        interaction_matrix[i][j] = val
        interaction_matrix[j][i] = val
    for (i, j), val in pair_cosine.items():
        cosine_matrix[i][j] = val
        cosine_matrix[j][i] = val
    for (i, j), val in pair_kl.items():
        kl_matrix[i][j] = val
        kl_matrix[j][i] = val

    result = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "n_pairs": len(pairs),
        "single_layer_kl": {str(k): v for k, v in single_kl.items()},
        "interaction_matrix": interaction_matrix,
        "cosine_matrix": cosine_matrix,
        "pair_kl_matrix": kl_matrix,
        "summary": {
            "n_subadditive": n_subadditive,
            "frac_subadditive": n_subadditive / len(pairs),
            "mean_interaction": mean_interaction,
            "median_interaction": float(np.median(interactions)),
            "mean_defect_cosine": mean_cosine,
            "median_defect_cosine": float(np.median(cosines)),
            "cos_vs_interaction_spearman": cos_vs_int_spearman.statistic,
            "cos_vs_interaction_spearman_p": cos_vs_int_spearman.pvalue,
            "cos_vs_interaction_pearson": cos_vs_int_pearson.statistic,
            "cos_vs_interaction_pearson_p": cos_vs_int_pearson.pvalue,
            "distance_vs_interaction_spearman": dist_vs_int_spearman.statistic,
            "distance_vs_interaction_spearman_p": dist_vs_int_spearman.pvalue,
            "distance_vs_cosine_spearman": dist_vs_cos_spearman.statistic,
            "distance_vs_cosine_spearman_p": dist_vs_cos_spearman.pvalue,
        },
        "pairs": [
            {
                "i": i,
                "j": j,
                "kl": pair_kl[(i, j)],
                "interaction": pair_interaction[(i, j)],
                "cosine": pair_cosine[(i, j)],
                "distance": abs(i - j),
            }
            for i, j in pairs
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
