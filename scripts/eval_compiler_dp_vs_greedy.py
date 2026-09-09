"""Compare greedy and DP graph compilers across quality budgets.

Tests whether the O(L log L) greedy compiler produces the same graph
as the O(L * resolution) DP solver under non-uniform latency savings.
The Blackwell profile is nearly uniform (greedy is provably optimal);
the MPS M4 Max profile has genuinely mixed savings and is the
interesting comparison.

Usage:
  python scripts/eval_compiler_dp_vs_greedy.py \
      --profile results/data/mps_m4max_graph.json \
      [--output paper_results/compiler/dp_vs_greedy.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fogen.execution_graph import (
    compile_graph_dp,
    compile_graph_greedy,
    single_layer_effects,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True,
                        help="Hardware graph profile JSON (with per-layer savings)")
    parser.add_argument("--graph_data", default=None,
                        help="Graph-rewrite JSON for quality costs (optional; "
                             "uses kl_effect from profile if available)")
    parser.add_argument("--n_budgets", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    layers = profile["per_layer_measurements"]["layers"]
    n_layers = len(layers)

    # Extract latency savings and quality costs
    saving_key = "saving_ms" if "saving_ms" in layers[0] else "fused_saving_ms"
    latency_savings = np.array([l[saving_key] for l in layers])
    # Clamp negative savings to 0 for the compiler (switching those layers is never beneficial)
    latency_savings_clamped = np.maximum(latency_savings, 0.0)

    # Quality costs: use kl_effect from profile if available, else defect
    if "kl_effect" in layers[0]:
        effect_costs = np.array([l["kl_effect"] for l in layers])
    elif "defect" in layers[0]:
        effect_costs = np.array([l["defect"] for l in layers])
    else:
        raise ValueError("Profile must have kl_effect or defect per layer")

    # If separate graph data provided, use its single-layer effects
    if args.graph_data:
        with open(args.graph_data) as f:
            gdata = json.load(f)
        effect_costs = single_layer_effects(gdata["rows"], "symmetric_kl")

    print(f"Hardware: {profile['hardware'].get('chip', profile['hardware'].get('gpu', '?'))}")
    print(f"Layers: {n_layers}")
    print(f"Latency savings range: [{latency_savings.min():.3f}, {latency_savings.max():.3f}] ms")
    print(f"Negative-saving layers: {(latency_savings < 0).sum()}/{n_layers}")
    print(f"Effect costs range: [{effect_costs.min():.4f}, {effect_costs.max():.4f}]")
    print()

    # Sweep budgets
    max_budget = float(effect_costs.sum()) * 1.2
    budgets = np.linspace(0.1, max_budget, args.n_budgets)

    results = []
    n_match = 0

    print(f"{'Budget':>8} {'Greedy':>8} {'DP':>8} {'Match':>6} "
          f"{'G_save':>8} {'D_save':>8} {'Regret':>8}")
    print("-" * 60)

    for budget in budgets:
        g = compile_graph_greedy(effect_costs, latency_savings_clamped, budget)
        d = compile_graph_dp(effect_costs, latency_savings_clamped, budget,
                             resolution=5000)

        g_par = sum(g["bits"])
        d_par = sum(d["bits"])
        match = g["bits"] == d["bits"]
        if match:
            n_match += 1

        g_save = g["predicted_saving"]
        d_save = d["predicted_saving"]
        regret = d_save - g_save  # positive = greedy is worse

        results.append({
            "budget": float(budget),
            "greedy_parallel": g_par,
            "dp_parallel": d_par,
            "match": bool(match),
            "greedy_saving_ms": float(g_save),
            "dp_saving_ms": float(d_save),
            "regret_ms": float(regret),
            "greedy_cost": float(g["predicted_effect"]),
            "dp_cost": float(d["predicted_effect"]),
        })

        print(f"{budget:>8.2f} {g_par:>8} {d_par:>8} {'  ✓' if match else '  ✗':>6} "
              f"{g_save:>8.3f} {d_save:>8.3f} {regret:>+8.4f}")

    match_frac = n_match / len(budgets)
    regrets = [r["regret_ms"] for r in results]
    max_regret = max(regrets)
    mean_regret = np.mean(regrets)

    # Recovered benefit: what fraction of DP's benefit does greedy capture?
    dp_benefits = [r["dp_saving_ms"] for r in results if r["dp_saving_ms"] > 0]
    g_benefits = [r["greedy_saving_ms"] for r in results if r["dp_saving_ms"] > 0]
    recovered = np.mean([g / d for g, d in zip(g_benefits, dp_benefits)
                         if d > 0]) if dp_benefits else 1.0

    print()
    print(f"Exact match: {n_match}/{len(budgets)} ({match_frac:.0%})")
    print(f"Max regret: {max_regret:+.4f} ms")
    print(f"Mean regret: {mean_regret:+.4f} ms")
    print(f"Recovered benefit: {recovered:.1%}")

    output = {
        "profile": args.profile,
        "n_layers": n_layers,
        "n_budgets": len(budgets),
        "exact_match_fraction": float(match_frac),
        "max_regret_ms": float(max_regret),
        "mean_regret_ms": float(mean_regret),
        "recovered_benefit": float(recovered),
        "budget_sweep": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
