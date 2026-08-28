"""Measure per-layer and end-to-end latency, then compile the optimal graph.

Compiles the optimal execution graph for the current hardware using
the defect-budget graph compiler with measured latency savings.
Supports CUDA, MPS, and CPU backends.

Usage:
    uv run python scripts/eval_mps_graph.py \
        --graph_data results/data/climbmix_graphs.json \
        --output results/data/blackwell_graph.json \
        --batch_size 1 --sequence_length 512

    # Force a specific device:
    uv run python scripts/eval_mps_graph.py \
        --graph_data results/data/climbmix_graphs.json \
        --output results/data/mps_graph.json \
        --device mps
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def build_model(cfg, device):
    d_model = cfg["d_model"]
    n_head = cfg["n_head"]
    head_dim = d_model // n_head
    n_layer = cfg["n_layer"]
    vocab_size = cfg["vocab_size"]

    class Attention(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.ve_lambdas = (
                nn.Parameter(torch.tensor([0.5, 0.5]))
                if has_ve(layer_idx, n_layer)
                else None
            )

        def forward(self, x, ve, cos, sin):
            B, T, C = x.shape
            q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
            k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
            v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)
            if self.ve_lambdas is not None and ve is not None:
                ve_r = ve.view(B, T, n_head, head_dim).transpose(1, 2)
                v = self.ve_lambdas[0] * v + self.ve_lambdas[1] * ve_r
            q1, q2 = q.chunk(2, dim=-1)
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
            q = F.rms_norm(q, (head_dim,))
            k = F.rms_norm(k, (head_dim,))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(y.transpose(1, 2).reshape(B, T, C))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(d_model, 4 * d_model, bias=False)
            self.down = nn.Linear(4 * d_model, d_model, bias=False)

        def forward(self, x):
            return self.down(F.relu(self.up(x)).square())

    class Block(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.attn = Attention(layer_idx)
            self.mlp = MLP()

        def forward_sequential(self, x, ve, cos, sin):
            n = F.rms_norm(x, (d_model,))
            a = self.attn(n, ve, cos, sin)
            x = x + a
            return x + self.mlp(F.rms_norm(x, (d_model,)))

        def forward_parallel(self, x, ve, cos, sin):
            n = F.rms_norm(x, (d_model,))
            a = self.attn(n, ve, cos, sin)
            return x + a + self.mlp(n)

        def forward(self, x, ve, cos, sin, mode="sequential"):
            if mode == "parallel":
                return self.forward_parallel(x, ve, cos, sin)
            return self.forward_sequential(x, ve, cos, sin)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList(
                [Block(i) for i in range(n_layer)]
            )
            self.value_embeds = nn.ModuleDict(
                {
                    str(i): nn.Embedding(vocab_size, d_model)
                    for i in range(n_layer)
                    if has_ve(i, n_layer)
                }
            )
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            half = head_dim // 2
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(half).float() / half)
            )
            t = torch.arange(cfg["ctx_len"]).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("rope_cos", freqs.cos()[None, None])
            self.register_buffer("rope_sin", freqs.sin()[None, None])

        def forward(self, idx, modes=None):
            T = idx.size(1)
            cos = self.rope_cos[:, :, :T]
            sin = self.rope_sin[:, :, :T]
            x = F.rms_norm(self.wte(idx), (d_model,))
            if modes is None:
                modes = ["sequential"] * n_layer
            for i, block in enumerate(self.blocks):
                ve = (
                    self.value_embeds[str(i)](idx)
                    if str(i) in self.value_embeds
                    else None
                )
                x = block(x, ve, cos, sin, mode=modes[i])
            logits = self.lm_head(F.rms_norm(x, (d_model,))).float()
            return 15.0 * torch.tanh(logits / 15.0)

    return Model().to(device).eval()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure_layer(block, x, ve, cos, sin, mode, device, warmup=20, repeats=100):
    fn = block.forward_sequential if mode == "sequential" else block.forward_parallel
    with torch.no_grad():
        for _ in range(warmup):
            fn(x, ve, cos, sin)
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(x, ve, cos, sin)
        synchronize(device)
    return (time.perf_counter() - t0) / repeats


def bench_full(model, idx, modes, device, warmup=10, repeats=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(idx, modes=modes)
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(idx, modes=modes)
        synchronize(device)
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None,
                        help="Force device (cuda/mps/cpu). Auto-detects if omitted.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    with open(args.graph_data) as f:
        graph_data = json.load(f)

    n_layer = len(graph_data["layer_defects"])
    cfg = {
        "vocab_size": 8192,
        "n_layer": n_layer,
        "d_model": 704,
        "n_head": 8,
        "ctx_len": 1024,
    }

    print(f"Building {cfg['n_layer']}-layer model on {device}...")
    model = build_model(cfg, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.1f}M")

    torch.manual_seed(2026)
    idx = torch.randint(
        0, cfg["vocab_size"], (args.batch_size, args.sequence_length), device=device
    )
    cos = model.rope_cos[:, :, : args.sequence_length]
    sin = model.rope_sin[:, :, : args.sequence_length]

    # Collect per-layer activations
    with torch.no_grad():
        x = F.rms_norm(model.wte(idx), (cfg["d_model"],))
        layer_inputs = []
        for i, block in enumerate(model.blocks):
            ve = (
                model.value_embeds[str(i)](idx)
                if str(i) in model.value_embeds
                else None
            )
            layer_inputs.append((x.clone(), ve))
            x = block.forward_sequential(x, ve, cos, sin)

    # Measure per-layer latency
    print(f"\nMeasuring per-layer latency (B={args.batch_size}, T={args.sequence_length})...")
    layer_rows = []
    for i, block in enumerate(model.blocks):
        x_i, ve_i = layer_inputs[i]
        t_seq = measure_layer(block, x_i, ve_i, cos, sin, "sequential",
                              device, args.warmup, args.repeats) * 1000
        t_par = measure_layer(block, x_i, ve_i, cos, sin, "parallel",
                              device, args.warmup, args.repeats) * 1000
        saving = t_seq - t_par
        layer_rows.append({
            "layer": i,
            "has_ve": has_ve(i, n_layer),
            "seq_ms": round(t_seq, 3),
            "par_ms": round(t_par, 3),
            "saving_ms": round(saving, 3),
            "defect": graph_data["layer_defects"][i],
            "kl_effect": None,
        })
        print(f"  Layer {i:2d}: seq={t_seq:.3f}ms  par={t_par:.3f}ms  saving={saving:+.3f}ms")

    # Compute KL effects from graph data
    rows = graph_data["rows"]
    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
    for r in rows:
        if sum(r["bits"]) == 1:
            layer = r["bits"].index(1)
            layer_rows[layer]["kl_effect"] = r["symmetric_kl"] - baseline_kl

    # Compile optimal graph
    import sys
    sys.path.insert(0, "src")
    from fogen.execution_graph import compile_graph, fit_composition_scale, single_layer_effects

    effects = single_layer_effects(rows, "symmetric_kl")
    multi = [r for r in rows if sum(r["bits"]) >= 2]
    predicted = [sum(effects[i] * b for i, b in enumerate(r["bits"])) for r in multi]
    actual = [r["symmetric_kl"] - baseline_kl for r in multi]
    alpha = fit_composition_scale(predicted, actual)

    hw_savings = np.array([row["saving_ms"] for row in layer_rows])
    result = compile_graph(effects, hw_savings, 999.0, alpha)
    optimal_mask = result["bits"]
    optimal_modes = ["parallel" if b else "sequential" for b in optimal_mask]

    print(f"\nOptimal graph: {[i for i, b in enumerate(optimal_mask) if b]}")
    print(f"Predicted KL: {result['predicted_effect']:.4f}")
    print(f"Predicted saving: {result['predicted_saving']:.3f}ms")

    # End-to-end benchmark
    print("\nEnd-to-end benchmarks...")
    t_seq = bench_full(model, idx, ["sequential"] * n_layer, device)
    t_par = bench_full(model, idx, ["parallel"] * n_layer, device)
    t_opt = bench_full(model, idx, optimal_modes, device)

    print(f"  Sequential:   {t_seq:.2f}ms")
    print(f"  All-parallel: {t_par:.2f}ms ({t_seq / t_par:.3f}x)")
    print(f"  Compiled:     {t_opt:.2f}ms ({t_seq / t_opt:.3f}x)")

    import platform
    hw_info = {"backend": device, "pytorch_version": torch.__version__,
               "os": platform.system()}
    if device == "cuda":
        hw_info["gpu"] = torch.cuda.get_device_name(0)
        hw_info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    elif device == "mps":
        hw_info["chip"] = "Apple Silicon"

    output = {
        "hardware": hw_info,
        "model": cfg,
        "optimal_graph": {
            "mask": optimal_mask,
            "modes": optimal_modes,
            "parallel_layers": [i for i, b in enumerate(optimal_mask) if b],
            "sequential_layers": [i for i, b in enumerate(optimal_mask) if not b],
            "n_parallel": sum(optimal_mask),
            "predicted_kl": result["predicted_effect"],
            "composition_scale": alpha,
        },
        "per_layer_measurements": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "layers": layer_rows,
        },
        "end_to_end_benchmarks": [
            {
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "all_sequential_ms": round(t_seq, 2),
                "all_parallel_ms": round(t_par, 2),
                "mps_compiled_ms": round(t_opt, 2),
                "speedup_vs_sequential": round(t_seq / t_opt, 3),
                "speedup_vs_all_parallel": round(t_par / t_opt, 3),
            }
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
