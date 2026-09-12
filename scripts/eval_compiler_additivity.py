"""Test whether per-layer latency savings are additive.

The graph compiler assumes total_saving ≈ Σ single-layer savings.
This script checks that assumption against measured multi-mask latencies.

Reports Pearson R², MAE, and whether the compiler's predicted-best
graph is near the actual-best.

Usage:
  python scripts/eval_compiler_additivity.py \
      --graph_data blackwell/results/430m_poly_graphs.json \
      [--profile blackwell/results/graph_profile_430m.json] \
      [--output results/compiler_additivity.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True,
                        help="Graph-rewrite JSON with latency_seconds per mask")
    parser.add_argument("--profile", default=None,
                        help="Optional graph profile JSON with per-layer timings")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        data = json.load(f)

    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    # --- Extract single-layer latency savings ---
    baseline_latency = None
    single_savings = np.zeros(n_layers)

    for row in rows:
        n_par = sum(row["bits"])
        lat = row.get("latency_seconds")
        if lat is None:
            continue
        if n_par == 0:
            baseline_latency = lat
        elif n_par == 1:
            layer = row["bits"].index(1)
            single_savings[layer] = baseline_latency - lat if baseline_latency else 0

    if baseline_latency is None:
        print("ERROR: no all-sequential baseline found")
        return

    print(f"Baseline latency: {baseline_latency*1000:.2f} ms")
    print(f"Layers: {n_layers}")
    print(f"Mean single-layer saving: {single_savings.mean()*1000:.3f} ms")
    print(f"Max single-layer saving: {single_savings.max()*1000:.3f} ms")
    print()

    # --- Compare additive prediction vs actual for multi-layer masks ---
    multi = [r for r in rows
             if sum(r["bits"]) > 1
             and r.get("latency_seconds") is not None]

    if not multi:
        print("No multi-layer masks with latency data.")
        return

    predicted_savings = []
    actual_savings = []
    n_parallel_list = []

    for row in multi:
        bits = np.array(row["bits"])
        pred = float(bits @ single_savings)
        actual = baseline_latency - row["latency_seconds"]
        predicted_savings.append(pred)
        actual_savings.append(actual)
        n_parallel_list.append(int(sum(row["bits"])))

    predicted_savings = np.array(predicted_savings)
    actual_savings = np.array(actual_savings)
    n_parallel_arr = np.array(n_parallel_list)

    r_pearson, p_pearson = pearsonr(predicted_savings, actual_savings)
    r_spearman, p_spearman = spearmanr(predicted_savings, actual_savings)
    mae = float(np.mean(np.abs(predicted_savings - actual_savings)))
    rmse = float(np.sqrt(np.mean((predicted_savings - actual_savings) ** 2)))
    r2 = r_pearson ** 2

    print(f"=== Latency additivity ({len(multi)} multi-layer masks) ===")
    print(f"  Pearson R²:  {r2:.4f}")
    print(f"  Pearson r:   {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  Spearman ρ:  {r_spearman:.4f} (p={p_spearman:.2e})")
    print(f"  MAE:         {mae*1000:.3f} ms")
    print(f"  RMSE:        {rmse*1000:.3f} ms")
    print(f"  Mean actual: {actual_savings.mean()*1000:.3f} ms")
    print()

    # --- Check by cardinality ---
    print("  By cardinality:")
    for k in sorted(set(n_parallel_list)):
        mask_k = n_parallel_arr == k
        if mask_k.sum() < 2:
            continue
        pred_k = predicted_savings[mask_k]
        act_k = actual_savings[mask_k]
        ratio = act_k.mean() / max(pred_k.mean(), 1e-12)
        print(f"    |m|={k:>2}: predicted={pred_k.mean()*1000:.3f}ms "
              f"actual={act_k.mean()*1000:.3f}ms ratio={ratio:.3f}")

    # --- Compiler's predicted-best vs actual-best ---
    # The compiler selects the mask maximizing predicted saving subject to a budget
    # Here we just check: does the mask with the highest predicted saving
    # also have the highest (or near-highest) actual saving?
    all_with_lat = [r for r in rows if r.get("latency_seconds") is not None]

    best_predicted_idx = max(range(len(all_with_lat)),
                            key=lambda i: float(np.array(all_with_lat[i]["bits"]) @ single_savings))
    best_actual_idx = max(range(len(all_with_lat)),
                          key=lambda i: baseline_latency - all_with_lat[i]["latency_seconds"])

    bp = all_with_lat[best_predicted_idx]
    ba = all_with_lat[best_actual_idx]

    bp_actual_saving = baseline_latency - bp["latency_seconds"]
    ba_actual_saving = baseline_latency - ba["latency_seconds"]

    print()
    print("=== Compiler optimality ===")
    print(f"  Predicted-best mask: {sum(bp['bits'])}/{n_layers} parallel, "
          f"actual saving={bp_actual_saving*1000:.2f}ms")
    print(f"  Actual-best mask:    {sum(ba['bits'])}/{n_layers} parallel, "
          f"actual saving={ba_actual_saving*1000:.2f}ms")
    if ba_actual_saving > 0:
        optimality = bp_actual_saving / ba_actual_saving
        print(f"  Compiler achieves {optimality:.1%} of actual-best saving")
    else:
        optimality = None
        print("  (actual-best saving is non-positive)")

    # --- Optional: compare with profile data ---
    if args.profile:
        with open(args.profile) as f:
            profile = json.load(f)
        layers = profile.get("per_layer_measurements", {}).get("layers", [])
        if layers:
            profile_savings = np.array([l.get("fused_saving_ms", l.get("par_saving_ms", 0))
                                        for l in layers]) / 1000  # ms -> s
            graph_savings = single_savings
            pr, _ = pearsonr(profile_savings, graph_savings)
            print()
            print(f"=== Profile vs graph-measured savings ===")
            print(f"  Pearson r: {pr:.4f}")
            print(f"  Profile mean: {profile_savings.mean()*1000:.3f}ms, "
                  f"Graph mean: {graph_savings.mean()*1000:.3f}ms")

    result = {
        "graph_data": args.graph_data,
        "n_layers": n_layers,
        "n_multi_masks": len(multi),
        "baseline_latency_ms": baseline_latency * 1000,
        "pearson_r2": r2,
        "pearson_r": float(r_pearson),
        "spearman_rho": float(r_spearman),
        "mae_ms": mae * 1000,
        "rmse_ms": rmse * 1000,
        "compiler_optimality": optimality,
        "mean_single_saving_ms": float(single_savings.mean() * 1000),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
