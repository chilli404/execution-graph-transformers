"""Measure per-layer and end-to-end latency, then compile the optimal graph.

Profiles sequential, parallel, and fused execution modes across batch sizes.
Compiles the optimal execution graph using the defect-budget graph compiler
with measured hardware-specific latency savings.

Supports CUDA, MPS, and CPU backends.

Usage:
    # 120M model on auto-detected device:
    uv run python scripts/eval_mps_graph.py \
        --graph_data results/data/climbmix_graphs.json \
        --output blackwell/results/graph_profile.json

    # 430M shape (no graph_data needed for pure timing):
    uv run python scripts/eval_mps_graph.py \
        --graph_data results/data/climbmix_graphs.json \
        --output blackwell/results/graph_profile_430m.json \
        --d_model 1152 --n_head 18 --n_layer 20

    # Specific device and batch sweep:
    uv run python scripts/eval_mps_graph.py \
        --graph_data results/data/climbmix_graphs.json \
        --output results.json \
        --device cuda --batch_sizes 1 4 8 16 32
"""

import argparse
import json
import platform
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
            self._fused_weight = None

        def forward_sequential(self, x, ve, cos, sin):
            n = F.rms_norm(x, (d_model,))
            a = self.attn(n, ve, cos, sin)
            x = x + a
            return x + self.mlp(F.rms_norm(x, (d_model,)))

        def forward_parallel(self, x, ve, cos, sin):
            n = F.rms_norm(x, (d_model,))
            a = self.attn(n, ve, cos, sin)
            return x + a + self.mlp(n)

        def forward_fused(self, x, ve, cos, sin):
            B, T, C = x.shape
            n = F.rms_norm(x, (d_model,))
            if self._fused_weight is None:
                self._fused_weight = torch.cat([
                    self.attn.wq.weight,
                    self.attn.wk.weight,
                    self.attn.wv.weight,
                    self.mlp.up.weight,
                ]).detach()
            projected = F.linear(n, self._fused_weight)
            q_raw, k_raw, v_raw, mlp_h = projected.split(
                [d_model, d_model, d_model, 4 * d_model], dim=-1)
            q = q_raw.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k_raw.view(B, T, n_head, head_dim).transpose(1, 2)
            v = v_raw.view(B, T, n_head, head_dim).transpose(1, 2)
            if self.attn.ve_lambdas is not None and ve is not None:
                ve_r = ve.view(B, T, n_head, head_dim).transpose(1, 2)
                v = self.attn.ve_lambdas[0] * v + self.attn.ve_lambdas[1] * ve_r
            q1, q2 = q.chunk(2, dim=-1)
            q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
            q = F.rms_norm(q, (head_dim,))
            k = F.rms_norm(k, (head_dim,))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = self.attn.wo(y.transpose(1, 2).reshape(B, T, C))
            m = self.mlp.down(F.relu(mlp_h).square())
            return x + a + m

        def forward(self, x, ve, cos, sin, mode="sequential"):
            if mode == "parallel":
                return self.forward_parallel(x, ve, cos, sin)
            if mode == "parallel_fused":
                return self.forward_fused(x, ve, cos, sin)
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
    fns = {
        "sequential": block.forward_sequential,
        "parallel": block.forward_parallel,
        "parallel_fused": block.forward_fused,
    }
    fn = fns[mode]
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
    parser = argparse.ArgumentParser(
        description="Profile execution graph latency and compile optimal graph")
    parser.add_argument("--graph_data", required=True,
                        help="Path to graph results JSON (e.g. climbmix_graphs.json)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--device", default=None,
                        help="Force device (cuda/mps/cpu). Auto-detects if omitted.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes for sweep")
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=None,
                        help="Override model width (default: from graph_data or 704)")
    parser.add_argument("--n_head", type=int, default=None,
                        help="Override head count")
    parser.add_argument("--n_layer", type=int, default=None,
                        help="Override layer count")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--verify-greedy", action="store_true",
                        help="Run brute-force compiler alongside greedy and verify match. "
                             "Feasible up to ~20 layers (takes ~1.5s at L=20).")
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

    n_layer = args.n_layer or len(graph_data["layer_defects"])
    cfg = {
        "vocab_size": 8192,
        "n_layer": n_layer,
        "d_model": args.d_model or 704,
        "n_head": args.n_head or 8,
        "ctx_len": max(1024, args.sequence_length),
    }

    print(f"Building model on {device}...")
    print(f"  layers={cfg['n_layer']}, d_model={cfg['d_model']}, "
          f"n_head={cfg['n_head']}, ctx={cfg['ctx_len']}")
    model = build_model(cfg, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params / 1e6:.1f}M")

    torch.manual_seed(2026)
    T = args.sequence_length
    primary_batch = args.batch_sizes[0]

    # --- Per-layer profiling at primary batch size ---
    idx = torch.randint(0, cfg["vocab_size"], (primary_batch, T), device=device)
    cos = model.rope_cos[:, :, :T]
    sin = model.rope_sin[:, :, :T]

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

    print(f"\nPer-layer latency (B={primary_batch}, T={T}):")
    print(f"  {'Layer':>5} | {'Seq':>8} | {'Par':>8} | {'Fused':>8} | {'Par save':>9} | {'Fused save':>10}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}")

    layer_rows = []
    for i, block in enumerate(model.blocks):
        x_i, ve_i = layer_inputs[i]
        t_seq = measure_layer(block, x_i, ve_i, cos, sin, "sequential",
                              device, args.warmup, args.repeats) * 1000
        t_par = measure_layer(block, x_i, ve_i, cos, sin, "parallel",
                              device, args.warmup, args.repeats) * 1000
        t_fused = measure_layer(block, x_i, ve_i, cos, sin, "parallel_fused",
                                device, args.warmup, args.repeats) * 1000
        layer_rows.append({
            "layer": i,
            "has_ve": has_ve(i, n_layer),
            "seq_ms": round(t_seq, 4),
            "par_ms": round(t_par, 4),
            "fused_ms": round(t_fused, 4),
            "par_saving_ms": round(t_seq - t_par, 4),
            "fused_saving_ms": round(t_seq - t_fused, 4),
        })
        print(f"  {i:>5} | {t_seq:>7.3f}ms | {t_par:>7.3f}ms | {t_fused:>7.3f}ms | "
              f"{t_seq - t_par:>+8.3f}ms | {t_seq - t_fused:>+9.3f}ms")

    # --- Batch size sweep (end-to-end) ---
    print(f"\nBatch size sweep (T={T}):")
    print(f"  {'B':>4} | {'Seq':>9} | {'Par':>9} | {'Fused':>9} | {'Par/Seq':>7} | {'Fused/Seq':>9}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}")

    batch_sweep = []
    for B in args.batch_sizes:
        try:
            idx_b = torch.randint(0, cfg["vocab_size"], (B, T), device=device)
            all_seq = ["sequential"] * n_layer
            all_par = ["parallel"] * n_layer
            all_fused = ["parallel_fused"] * n_layer

            t_s = bench_full(model, idx_b, all_seq, device, args.warmup // 2, args.repeats // 2)
            t_p = bench_full(model, idx_b, all_par, device, args.warmup // 2, args.repeats // 2)
            t_f = bench_full(model, idx_b, all_fused, device, args.warmup // 2, args.repeats // 2)

            batch_sweep.append({
                "batch_size": B,
                "sequence_length": T,
                "sequential_ms": round(t_s, 3),
                "parallel_ms": round(t_p, 3),
                "fused_ms": round(t_f, 3),
                "parallel_speedup": round(t_s / t_p, 4),
                "fused_speedup": round(t_s / t_f, 4),
            })
            print(f"  {B:>4} | {t_s:>8.2f}ms | {t_p:>8.2f}ms | {t_f:>8.2f}ms | "
                  f"{t_s / t_p:>6.3f}x | {t_s / t_f:>8.3f}x")
        except RuntimeError as e:
            print(f"  {B:>4} | OOM: {e}")
            break

    # --- Compile optimal graph using defect data ---
    # Only possible if graph_data has matching layer count
    compiled_graph = None
    if len(graph_data["layer_defects"]) == n_layer:
        import sys
        sys.path.insert(0, "src")
        from fogen.execution_graph import (
            compile_graph_dp, compile_graph_greedy,
            fit_composition_scale, single_layer_effects,
        )

        rows = graph_data["rows"]
        baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)

        for r in rows:
            if sum(r["bits"]) == 1:
                layer = r["bits"].index(1)
                layer_rows[layer]["kl_effect"] = r["symmetric_kl"] - baseline_kl

        effects = single_layer_effects(rows, "symmetric_kl")
        multi = [r for r in rows if sum(r["bits"]) >= 2]
        predicted = [sum(effects[i] * b for i, b in enumerate(r["bits"])) for r in multi]
        actual = [r["symmetric_kl"] - baseline_kl for r in multi]
        alpha = fit_composition_scale(predicted, actual)

        # Use best available savings (fused if it helps, else parallel)
        par_savings = np.array([row["par_saving_ms"] for row in layer_rows])
        fused_savings = np.array([row["fused_saving_ms"] for row in layer_rows])
        best_savings = np.maximum(par_savings, fused_savings)

        # DP is exact for non-uniform savings; greedy is a fast approximation
        result = compile_graph_dp(effects, best_savings, 999.0, alpha)
        optimal_mask = result["bits"]
        optimal_modes = ["parallel" if b else "sequential" for b in optimal_mask]

        # Choose fused for layers where fused > parallel
        for i in range(n_layer):
            if optimal_mask[i] and fused_savings[i] > par_savings[i]:
                optimal_modes[i] = "parallel_fused"

        compiled_graph = {
            "mask": optimal_mask,
            "modes": optimal_modes,
            "parallel_layers": [i for i, b in enumerate(optimal_mask) if b],
            "sequential_layers": [i for i, b in enumerate(optimal_mask) if not b],
            "n_parallel": sum(optimal_mask),
            "predicted_kl": result["predicted_effect"],
            "predicted_saving_ms": result["predicted_saving"],
            "composition_scale": alpha,
        }

        print(f"\nCompiled graph: {compiled_graph['parallel_layers']}")
        print(f"  Modes: {optimal_modes}")
        print(f"  Predicted KL: {result['predicted_effect']:.4f}")
        print(f"  Predicted saving: {result['predicted_saving']:.3f}ms")

        # Benchmark compiled graph
        idx_c = torch.randint(0, cfg["vocab_size"], (primary_batch, T), device=device)
        t_compiled = bench_full(model, idx_c, optimal_modes, device)
        t_seq_c = bench_full(model, idx_c, ["sequential"] * n_layer, device)
        compiled_graph["measured_sequential_ms"] = round(t_seq_c, 3)
        compiled_graph["measured_compiled_ms"] = round(t_compiled, 3)
        compiled_graph["measured_speedup"] = round(t_seq_c / t_compiled, 4)
        print(f"  Measured: seq={t_seq_c:.2f}ms, compiled={t_compiled:.2f}ms "
              f"({t_seq_c / t_compiled:.3f}x)")
    else:
        print(f"\nGraph data has {len(graph_data['layer_defects'])} layers, "
              f"model has {n_layer} — using timing-only compilation")
        # Without defect data for this layer count, compile using timing alone:
        # parallelize all layers where parallel/fused is faster (no quality budget)
        par_savings = np.array([row["par_saving_ms"] for row in layer_rows])
        fused_savings = np.array([row["fused_saving_ms"] for row in layer_rows])
        best_savings = np.maximum(par_savings, fused_savings)

        optimal_mask = [1 if best_savings[i] > 0.01 else 0 for i in range(n_layer)]
        optimal_modes = []
        for i in range(n_layer):
            if not optimal_mask[i]:
                optimal_modes.append("sequential")
            elif fused_savings[i] > par_savings[i]:
                optimal_modes.append("parallel_fused")
            else:
                optimal_modes.append("parallel")

        compiled_graph = {
            "mask": optimal_mask,
            "modes": optimal_modes,
            "parallel_layers": [i for i, b in enumerate(optimal_mask) if b],
            "sequential_layers": [i for i, b in enumerate(optimal_mask) if not b],
            "n_parallel": sum(optimal_mask),
            "method": "timing_only",
            "note": "No defect data for this layer count; selected by latency gain only",
        }

        print(f"\nTiming-only compiled graph: {compiled_graph['parallel_layers']}")
        print(f"  Modes: {optimal_modes}")
        total_saving = sum(best_savings[i] for i in range(n_layer) if optimal_mask[i])
        print(f"  Total predicted saving: {total_saving:.3f}ms")

        idx_c = torch.randint(0, cfg["vocab_size"], (primary_batch, T), device=device)
        t_compiled = bench_full(model, idx_c, optimal_modes, device)
        t_seq_c = bench_full(model, idx_c, ["sequential"] * n_layer, device)
        compiled_graph["measured_sequential_ms"] = round(t_seq_c, 3)
        compiled_graph["measured_compiled_ms"] = round(t_compiled, 3)
        compiled_graph["measured_speedup"] = round(t_seq_c / t_compiled, 4)
        print(f"  Measured: seq={t_seq_c:.2f}ms, compiled={t_compiled:.2f}ms "
              f"({t_seq_c / t_compiled:.3f}x)")

    # --- Verify greedy vs brute-force ---
    greedy_verification = None
    if getattr(args, "verify_greedy", False) and compiled_graph and n_layer <= 24:
        import sys
        if "src" not in sys.path:
            sys.path.insert(0, "src")
        from fogen.execution_graph import compile_graph, compile_graph_greedy

        par_savings = np.array([row["par_saving_ms"] for row in layer_rows])
        fused_savings = np.array([row["fused_saving_ms"] for row in layer_rows])
        best_savings = np.maximum(par_savings, fused_savings)

        if len(graph_data["layer_defects"]) == n_layer:
            # Has defect data — compare with quality-budgeted compilation
            effects = single_layer_effects(rows, "symmetric_kl")
            budgets = [1.0, 2.0, 3.0, 5.0, 999.0]
            print(f"\n--- Greedy vs brute-force verification (L={n_layer}) ---")
            import time as _time
            t_bf_total, t_gr_total = 0, 0
            matches = 0
            for budget in budgets:
                t0 = _time.perf_counter()
                bf = compile_graph(effects, best_savings, budget, alpha)
                t_bf = _time.perf_counter() - t0

                t0 = _time.perf_counter()
                gr = compile_graph_greedy(effects, best_savings, budget, alpha)
                t_gr = _time.perf_counter() - t0

                t_bf_total += t_bf
                t_gr_total += t_gr
                match = bf["predicted_saving"] == gr["predicted_saving"]
                if match:
                    matches += 1
                status = "MATCH" if match else "MISMATCH"
                print(f"  Budget={budget:.1f}: BF={bf['predicted_saving']:.3f}ms "
                      f"({sum(bf['bits'])} layers, {t_bf:.3f}s) | "
                      f"Greedy={gr['predicted_saving']:.3f}ms "
                      f"({sum(gr['bits'])} layers, {t_gr*1000:.3f}ms) | {status}")

            greedy_verification = {
                "n_layers": n_layer,
                "budgets_tested": budgets,
                "all_match": matches == len(budgets),
                "matches": matches,
                "total": len(budgets),
                "brute_force_time_s": round(t_bf_total, 3),
                "greedy_time_s": round(t_gr_total, 6),
                "speedup": round(t_bf_total / max(t_gr_total, 1e-9), 0),
            }
            print(f"  Result: {matches}/{len(budgets)} match | "
                  f"BF: {t_bf_total:.3f}s, Greedy: {t_gr_total*1000:.3f}ms "
                  f"({t_bf_total/max(t_gr_total, 1e-9):.0f}x faster)")
        else:
            # Timing-only: uniform savings, greedy is provably optimal
            # Still verify to be safe
            uniform_savings = np.ones(n_layer)
            # Use synthetic costs for verification
            costs = best_savings.copy()
            costs[costs <= 0] = 0.001
            print(f"\n--- Greedy vs brute-force verification (L={n_layer}, timing-only) ---")
            import time as _time
            t0 = _time.perf_counter()
            bf = compile_graph(costs, uniform_savings, 999.0, 1.0)
            t_bf = _time.perf_counter() - t0
            t0 = _time.perf_counter()
            gr = compile_graph_greedy(costs, uniform_savings, 999.0, 1.0)
            t_gr = _time.perf_counter() - t0
            match = bf["bits"] == gr["bits"]
            print(f"  BF: {sum(bf['bits'])} layers in {t_bf:.3f}s | "
                  f"Greedy: {sum(gr['bits'])} layers in {t_gr*1000:.3f}ms | "
                  f"{'MATCH' if match else 'MISMATCH'}")
            print(f"  Speedup: {t_bf/max(t_gr, 1e-9):.0f}x")
            greedy_verification = {
                "n_layers": n_layer,
                "method": "timing_only_uniform",
                "match": match,
                "brute_force_time_s": round(t_bf, 3),
                "greedy_time_s": round(t_gr, 6),
                "speedup": round(t_bf / max(t_gr, 1e-9), 0),
            }

    # --- Output ---
    hw_info = {"backend": device, "pytorch_version": torch.__version__,
               "platform": platform.system()}
    if device == "cuda":
        hw_info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        hw_info["gpu_memory_gb"] = round(mem / 1e9, 1)
    elif device == "mps":
        hw_info["chip"] = "Apple Silicon"

    output = {
        "hardware": hw_info,
        "model": cfg,
        "model_params_m": round(params / 1e6, 1),
        "per_layer_measurements": {
            "batch_size": primary_batch,
            "sequence_length": T,
            "layers": layer_rows,
        },
        "batch_sweep": batch_sweep,
    }
    if compiled_graph:
        output["compiled_graph"] = compiled_graph
    if greedy_verification:
        output["greedy_verification"] = greedy_verification

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
