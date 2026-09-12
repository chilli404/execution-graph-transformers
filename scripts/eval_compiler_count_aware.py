"""Compare count-aware exact DP against greedy compiler.

The composition law is D(m) = alpha0 * sum(d_l) * |m|^(-beta). The cost depends on
both which layers are selected AND how many, making it a cardinality-
constrained knapsack. This script implements exact DP[layer, budget, count]
and compares against the greedy approximation.

Usage:
  python scripts/eval_compiler_count_aware.py \
      --graph_data paper_results/core_scaling/430m_poly_graphs.json \
      --output paper_results/compiler/count_aware_comparison.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from fogen.execution_graph import (compile_graph_greedy, fit_count_adjusted_scale,
                                    single_layer_effects)

def compile_count_aware_dp(effect_costs, latency_savings, budget, alpha0, beta,
                           resolution=500):
    n = len(effect_costs)
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    max_sum = float(effect_costs.sum())
    if max_sum <= 0:
        return {"bits": [0]*n, "predicted_effect": 0.0, "predicted_saving": 0.0, "count": 0}
    step = max_sum / resolution
    dp = {0: (0.0, 0, 0.0, tuple([0]*n))}
    for i in range(n):
        if latency_savings[i] <= 0:
            continue
        int_cost = max(1, int(round(effect_costs[i] / step)))
        new_dp = {}
        for r, (sav, cnt, esum, bits) in dp.items():
            if r not in new_dp or sav > new_dp[r][0]:
                new_dp[r] = (sav, cnt, esum, bits)
            new_r = r + int_cost
            if new_r <= resolution:
                nb = list(bits); nb[i] = 1
                entry = (sav + latency_savings[i], cnt + 1, esum + effect_costs[i], tuple(nb))
                if new_r not in new_dp or entry[0] > new_dp[new_r][0]:
                    new_dp[new_r] = entry
        dp = new_dp
    best = None
    for r, (sav, cnt, esum, bits) in dp.items():
        cost = alpha0 * esum * (cnt ** (-beta)) if cnt > 0 else 0.0
        if cost <= budget + 1e-9:
            if best is None or sav > best[0]:
                best = (sav, cnt, esum, cost, list(bits))
    if best is None:
        return {"bits": [0]*n, "predicted_effect": 0.0, "predicted_saving": 0.0, "count": 0}
    return {"bits": best[4], "predicted_effect": float(best[3]),
            "predicted_saving": float(best[0]), "count": int(best[1])}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_budgets", type=int, default=15)
    args = parser.parse_args()
    with open(args.graph_data) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])
    kl_effects = single_layer_effects(rows, "symmetric_kl")
    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
    multi = [r for r in rows if sum(r["bits"]) > 1]
    pred_sums = np.array([float(np.dot(r["bits"], kl_effects)) for r in multi])
    actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
    n_par = np.array([sum(r["bits"]) for r in multi], dtype=float)
    alpha0, beta = fit_count_adjusted_scale(pred_sums, actual_kl, n_par)
    print(f"Composition: alpha0={alpha0:.4f}, beta={beta:.4f}")
    uniform_savings = np.ones(n_layers)
    kl_range = actual_kl[actual_kl > 0]
    budgets = np.linspace(float(np.percentile(kl_range, 5)),
                          float(np.percentile(kl_range, 95)), args.n_budgets)
    print(f"\n{'Budget':>8} {'Greedy':>7} {'DP':>4} {'Match':>6} {'Regret':>8}")
    print("-" * 40)
    comparisons = []
    for budget in budgets:
        greedy = compile_graph_greedy(kl_effects, uniform_savings, budget, scale=alpha0)
        dp = compile_count_aware_dp(kl_effects, uniform_savings, budget, alpha0, beta)
        match = greedy["bits"] == dp["bits"]
        regret = dp["predicted_saving"] - greedy["predicted_saving"]
        comparisons.append({"budget": float(budget), "greedy_k": sum(greedy["bits"]),
            "dp_k": dp["count"], "match": bool(match), "regret": float(regret),
            "greedy_saving": greedy["predicted_saving"], "dp_saving": dp["predicted_saving"]})
        print(f"{budget:>8.3f} {sum(greedy['bits']):>7} {dp['count']:>4} "
              f"{'Y' if match else 'N':>6} {regret:>+8.2f}")
    n_match = sum(1 for c in comparisons if c["match"])
    print(f"\nMatch: {n_match}/{len(comparisons)}, max regret: {max(c['regret'] for c in comparisons):.2f}")
    result = {"alpha0": alpha0, "beta": beta, "n_match": n_match,
              "match_fraction": n_match/len(comparisons),
              "max_regret": float(max(c["regret"] for c in comparisons)),
              "comparisons": comparisons}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
