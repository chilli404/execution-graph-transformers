"""Calibration sample efficiency: how many single-layer probes does the
composition predictor need?

For k = 1..L measured layers, estimates prediction quality using mean
imputation for unmeasured layers. Reports correlation and RMSE vs k.

Usage:
  python scripts/eval_calibration_sample_efficiency.py \
      --graph_data paper_results/core_scaling/430m_poly_graphs.json \
      --output paper_results/composition/sample_efficiency.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        data = json.load(f)

    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    baseline_kl = next(
        r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)

    # Extract single-layer effects
    single_kl = np.zeros(n_layers)
    for r in rows:
        if sum(r["bits"]) == 1:
            layer = r["bits"].index(1)
            single_kl[layer] = max(0, r["symmetric_kl"] - baseline_kl)

    # Multi-layer masks
    multi = [r for r in rows if sum(r["bits"]) > 1]
    bits_matrix = np.array([r["bits"] for r in multi])
    actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in multi])

    rng = np.random.default_rng(42)
    results_by_k = []

    for k in range(1, n_layers + 1):
        trial_pearsons = []
        trial_spearmans = []
        trial_rmses = []

        for _ in range(args.n_trials):
            measured = rng.choice(n_layers, size=k, replace=False)
            measured_set = set(measured)
            mean_measured = single_kl[measured].mean()

            # Imputed effects: measured layers use true value, others use mean
            imputed = np.full(n_layers, mean_measured)
            imputed[measured] = single_kl[measured]

            # Predict KL for all multi-layer masks
            pred_kl = bits_matrix @ imputed

            valid = actual_kl > 0
            if valid.sum() < 3:
                continue

            r_p = pearsonr(pred_kl[valid], actual_kl[valid]).statistic
            r_s = spearmanr(pred_kl[valid], actual_kl[valid]).statistic
            rmse = float(np.sqrt(np.mean((pred_kl[valid] - actual_kl[valid]) ** 2)))

            trial_pearsons.append(r_p)
            trial_spearmans.append(r_s)
            trial_rmses.append(rmse)

        results_by_k.append({
            "k": k,
            "fraction_measured": k / n_layers,
            "pearson_mean": float(np.mean(trial_pearsons)),
            "pearson_std": float(np.std(trial_pearsons)),
            "spearman_mean": float(np.mean(trial_spearmans)),
            "spearman_std": float(np.std(trial_spearmans)),
            "rmse_mean": float(np.mean(trial_rmses)),
            "rmse_std": float(np.std(trial_rmses)),
        })

        print(f"  k={k:>2}/{n_layers}  Pearson={np.mean(trial_pearsons):.3f}±{np.std(trial_pearsons):.3f}  "
              f"RMSE={np.mean(trial_rmses):.3f}±{np.std(trial_rmses):.3f}")

    # Find minimum k for 90% and 95% of full-L Pearson
    full_pearson = results_by_k[-1]["pearson_mean"]
    k_90 = next((r["k"] for r in results_by_k
                 if r["pearson_mean"] >= 0.90 * full_pearson), n_layers)
    k_95 = next((r["k"] for r in results_by_k
                 if r["pearson_mean"] >= 0.95 * full_pearson), n_layers)

    output = {
        "graph_data": args.graph_data,
        "n_layers": n_layers,
        "n_multi_masks": len(multi),
        "n_trials": args.n_trials,
        "full_pearson": full_pearson,
        "k_for_90pct_pearson": k_90,
        "k_for_95pct_pearson": k_95,
        "results_by_k": results_by_k,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull-L Pearson: {full_pearson:.3f}")
    print(f"90% of full Pearson at k={k_90} ({k_90/n_layers:.0%} of layers)")
    print(f"95% of full Pearson at k={k_95} ({k_95/n_layers:.0%} of layers)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
