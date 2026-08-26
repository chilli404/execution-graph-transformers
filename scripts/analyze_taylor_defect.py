import argparse
import json

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fogen.evals.bpb import val_stream
from fogen.model import GPT, ModelConfig, _rmsnorm


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def tensor_metrics(actual, approximation):
    actual = actual.float().reshape(-1)
    approximation = approximation.float().reshape(-1)
    actual_norm = actual.norm().clamp_min(1e-12)
    approximation_norm = approximation.norm()
    return {
        "cosine": float(torch.dot(actual, approximation) /
                        (actual_norm * approximation_norm.clamp_min(1e-12))),
        "relative_error": float((actual - approximation).norm() / actual_norm),
        "norm_ratio": float(approximation_norm / actual_norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--token_shards", required=True)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, args.config, device)
    stream = val_stream(args.token_shards)
    tokens = torch.tensor(
        np.asarray(stream[:args.sequence_length], dtype=np.int64)[None],
        device=device,
    )
    length = tokens.size(1)
    cos = model.rope_cos[:, :, :length]
    sin = model.rope_sin[:, :, :length]
    x = _rmsnorm(model.wte(tokens))
    rows = []
    for layer, block in enumerate(model.blocks):
        ve = model.value_embeds[str(layer)](tokens) if str(layer) in model.value_embeds else None
        normalized = _rmsnorm(x)
        attention = block.attn(normalized, ve, cos, sin)
        parallel_mlp = block.mlp(normalized)
        attended = x + attention
        sequential_mlp = block.mlp(_rmsnorm(attended))
        actual = sequential_mlp - parallel_mlp
        _, first_order = torch.func.jvp(
            lambda residual: block.mlp(_rmsnorm(residual)),
            (x,), (attention,))
        rows.append({"layer": layer, **tensor_metrics(actual, first_order)})
        x = attended + sequential_mlp
    result = {
        "checkpoint": args.ckpt,
        "sequence_length": args.sequence_length,
        "rows": rows,
        "mean_cosine": sum(row["cosine"] for row in rows) / len(rows),
        "mean_relative_error": sum(row["relative_error"] for row in rows) / len(rows),
        "mean_norm_ratio": sum(row["norm_ratio"] for row in rows) / len(rows),
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
