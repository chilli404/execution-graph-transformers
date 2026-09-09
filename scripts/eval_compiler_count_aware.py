"""Compare greedy compiler (additive budget) against count-aware solver.

The greedy compiler uses: max Σ v_l m_l  s.t. α Σ d_l m_l ≤ ε
The count-aware model uses: D(m) ≈ α₀ (Σ d_l m_l) |m|^(-β)

For each budget ε, the count-aware solver enumerates cardinalities k=1..L
and for each k finds the best k layers subject to the adjusted budget.

Usage:
  python scripts/eval_compiler_count_aware.py \
      --graph_data paper_results/core_scaling/430m_poly_graphs.json \
      --profile paper_results/compiler/graph_profile_430m.json \
      [--mps_profile paper_results/compiler/graph_profile.json] \
      --output paper_results/compiler/count_aware_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fogen.execution_graph import (
    compile_graph_greedy,
    fit_count_adjusted_scale,
    single_layer_effects,
)


def count_aware_compile(effect_costs, latency_savings, budget, alpha0, beta):
    """For each cardinality k, find the best k layers under the CA budget."""
    n = len(effect_costs)
    best = None

    for k in range(1, n + 1):
        # Under CA model: D(m) = α₀ * (Σ d_l) * k^(-β)
        # Budget constraint: α₀ * (Σ d_l) * k^(-β) ≤ ε
        # => Σ d_l ≤ ε / (α₀ * k^(-β))
        adjusted_budget = budget / (alpha0 * k ** (-beta))

        # For this k, select layers that maximize savings subject to Σ d_l ≤ adjusted_budget
        # Sort by savings/cost ratio (greedy for each k)
        ratios = np.where(effect_costs > 1e-12,
                          latency_savings / effect_costs, np.inf)
        order = np.argsort(-ratios)

        bits = [0] * n
        total_cost = 0.0
        count = 0
        for layer in order:
            if count >= k:
                break
            if latency_savings[layer] <= 0:
                continue
            new_cost = total_cost + effect_costs[layer]
            if new_cost <= adjusted_budget + 1e-12:
                bits[int(layer)] = 1
                total_cost = new_cost
                count += 1

        if count == 0:
            continue

        saving = float(np.dot(bits, latency_savings))
        predicted_kl = alpha0 * total_cost * count ** (-beta)

        candidate = {
            "bits": bits,
            "k": count,
            "predicted_effect_ca": predicted_kl,
            "predicted_saving": saving,
            "sum_d": total_cost,
        }

        if predicted_kl <= budget + 1e-12:
            if best is None or candidate["predicted_saving"] > best["predicted_saving"]:
                best = candidate

    if best is None:
        return {"bits": [0] * n, "k": 0, "predicted_effect_ca": 0.0,
                "predicted_saving": 0.0, "sum_d": 0.0}
    return best


def run_comparison(effect_costs, latency_savings, alpha0, beta, n_budgets=20):
    max_budget = alpha0 * float(np.sum(effect_costs))
    budgets = np.linspace(max_budget * 0.05, max_budget * 0.95, n_budgets)
    # Also use additive scale from the linear model
    linear_scale = 1.0

    results = []
    for eps in budgets:
        greedy = compile_graph_greedy(effect_costs, latency_savings, eps, scale=linear_scale)
        ca = count_aware_compile(effect_costs, latency_savings, eps, alpha0, beta)

        same = greedy["bits"] == ca["bits"]
        greedy_saving = greedy["predicted_saving"]
        ca_saving = ca["predicted_saving"]
        regret = ca_saving - greedy_saving if ca_saving > greedy_saving else 0.0

        results.append({
            "budget": float(eps),
            "greedy_n_parallel": sum(greedy["bits"]),
            "ca_n_parallel": ca["k"],
            "greedy_saving": greedy_saving,
            "ca_saving": ca_saving,
            "same_graph": same,
            "regret": regret,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--mps_profile", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load quality costs from graph data
    with open(args.graph_data) as f:
        gdata = json.load(f)

    kl_effects = single_layer_effects(gdata["rows"], "symmetric_kl")
    n_layers = len(kl_effects)

    # Fit CA model
    multi = [r for r in gdata["rows"] if 1 < sum(r["bits"]) < n_layers]
    baseline_kl = next(r["symmetric_kl"] for r in gdata["rows"] if sum(r["bits"]) == 0)
    pred_sums = [float(np.dot(r["bits"], kl_effects)) for r in multi]
    actual_kl = [r["symmetric_kl"] - baseline_kl for r in multi]
    n_par = [sum(r["bits"]) for r in multi]
    alpha0, beta = fit_count_adjusted_scale(pred_sums, actual_kl, n_par)

    print(f"Fitted CA model: α₀={alpha0:.4f}, β={beta:.4f}")
    print(f"Layers: {n_layers}")

    all_results = {}

    # Blackwell profile
    with open(args.profile) as f:
        profile = json.load(f)
    plm = profile["per_layer_measurements"]
    layers = plm["layers"]
    latency_savings = np.array([l.get("fused_saving_ms", l.get("par_saving_ms", 0))
                                for l in layers[:n_layers]])

    print(f"\n=== Blackwell (430M) ===")
    print(f"Latency savings range: {latency_savings.min():.4f} - {latency_savings.max():.4f} ms")
    bw_results = run_comparison(kl_effects, latency_savings, alpha0, beta)

    n_same = sum(r["same_graph"] for r in bw_results)
    max_regret = max(r["regret"] for r in bw_results)
    print(f"Same graph: {n_same}/{len(bw_results)}")
    print(f"Max regret: {max_regret:.6f} ms")
    all_results["blackwell"] = {
        "n_budgets": len(bw_results),
        "n_same": n_same,
        "frac_same": n_same / len(bw_results),
        "max_regret_ms": max_regret,
        "results": bw_results,
    }

    # MPS profile (non-uniform savings — more interesting)
    if args.mps_profile:
        with open(args.mps_profile) as f:
            mps = json.load(f)
        mps_layers = mps["per_layer_measurements"]["layers"]
        mps_n = len(mps_layers)
        mps_savings = np.array([l["saving_ms"] for l in mps_layers])
        mps_kl = np.array([l["kl_effect"] for l in mps_layers])

        # Fit CA on MPS single-layer KL effects (approximate)
        # Use the graph_data effects but scaled to MPS layer count if different
        if mps_n == n_layers:
            mps_effects = kl_effects
        else:
            # Use MPS-native KL effects
            mps_effects = mps_kl

        # Re-fit for MPS layer count if needed
        if mps_n != n_layers:
            mps_alpha0, mps_beta = alpha0, beta  # approximate
        else:
            mps_alpha0, mps_beta = alpha0, beta

        print(f"\n=== MPS M4 Max ===")
        print(f"Layers: {mps_n}")
        print(f"Latency savings range: {mps_savings.min():.4f} - {mps_savings.max():.4f} ms")

        mps_results = run_comparison(mps_effects, mps_savings, mps_alpha0, mps_beta)
        n_same_mps = sum(r["same_graph"] for r in mps_results)
        max_regret_mps = max(r["regret"] for r in mps_results)
        print(f"Same graph: {n_same_mps}/{len(mps_results)}")
        print(f"Max regret: {max_regret_mps:.6f} ms")

        all_results["mps"] = {
            "n_budgets": len(mps_results),
            "n_same": n_same_mps,
            "frac_same": n_same_mps / len(mps_results),
            "max_regret_ms": max_regret_mps,
            "results": mps_results,
        }

    # Summary
    print(f"\n=== Summary ===")
    print(f"{'Hardware':<15} {'Same':>6} {'Max regret':>12}")
    print("-" * 35)
    for hw, r in all_results.items():
        print(f"{hw:<15} {r['frac_same']:.0%}  {r['max_regret_ms']:>10.4f} ms")

    output = {
        "alpha0": alpha0,
        "beta": beta,
        "n_layers": n_layers,
        "hardware_results": all_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
