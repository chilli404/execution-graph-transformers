"""Mixed-mask calibration-size curve.

Measures how many multi-layer mask observations are needed to calibrate
the composition predictor. For each calibration size k, randomly samples
k masks (including singletons) for fitting and tests on the remainder.

Usage:
  python scripts/eval_calibration_size_curve.py \
      --graph_data paper_results/core_scaling/430m_poly_graphs.json \
      --output paper_results/composition/calibration_size_curve.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from fogen.execution_graph import fit_count_adjusted_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_repeats", type=int, default=20)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)

    # All non-baseline masks (singletons + multi-layer)
    all_masks = [r for r in rows if sum(r["bits"]) > 0]
    n_total = len(all_masks)

    # Pre-compute arrays
    bits_arr = np.array([r["bits"] for r in all_masks])
    actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in all_masks])
    n_par = bits_arr.sum(axis=1).astype(float)

    # Identify singletons (needed for single_layer_effects in every split)
    singleton_idx = {
        int(np.argmax(bits_arr[i])): i
        for i in range(n_total)
        if int(bits_arr[i].sum()) == 1
    }

    n_singletons = len(singleton_idx)
    # k = total calibration masks INCLUDING the mandatory singletons
    # So multi-layer calibration budget = k - n_singletons
    cal_sizes = [n_singletons + m for m in [3, 5, 10, 20, 30, 50, 75]]
    cal_sizes = [k for k in cal_sizes if k < n_total // 2]

    rng = np.random.default_rng(42)
    results = []

    print(f"Total masks: {n_total} (including {len(singleton_idx)} singletons)")
    print(f"{'k':>5} {'Pearson':>10} {'RMSE':>10} {'Viol_raw':>10} {'Viol_95':>10}")
    print("-" * 50)

    for k in cal_sizes:
        pearson_runs = []
        rmse_runs = []
        viol_raw_runs = []
        viol_95_runs = []

        for _ in range(args.n_repeats):
            # Always include singletons in calibration (needed for effects)
            singleton_indices = list(singleton_idx.values())
            remaining = [i for i in range(n_total) if i not in singleton_indices]

            # Sample additional masks to reach k total
            n_extra = max(0, k - len(singleton_indices))
            if n_extra > len(remaining):
                n_extra = len(remaining)
            extra = rng.choice(remaining, size=n_extra, replace=False).tolist()

            cal_idx = set(singleton_indices + extra)
            test_idx = [i for i in range(n_total) if i not in cal_idx]

            if len(test_idx) < 5:
                continue

            # Extract single-layer effects from calibration singletons
            effects = np.zeros(n_layers)
            for layer, idx in singleton_idx.items():
                effects[layer] = max(0, actual_kl[idx])

            # Fit on calibration multi-layer masks
            cal_multi = [i for i in cal_idx if n_par[i] > 1]
            if len(cal_multi) < 3:
                continue

            cal_pred_sums = np.array([float(bits_arr[i] @ effects) for i in cal_multi])
            cal_actual = actual_kl[cal_multi]
            cal_npar = n_par[cal_multi]

            try:
                alpha0, beta = fit_count_adjusted_scale(
                    cal_pred_sums, cal_actual, cal_npar)
            except Exception:
                continue

            # Predict on test set
            test_pred_sums = np.array([float(bits_arr[i] @ effects) for i in test_idx])
            test_npar = n_par[test_idx]
            test_actual = actual_kl[test_idx]

            test_predicted = alpha0 * test_pred_sums * np.power(
                np.maximum(test_npar, 1.0), -beta)

            # Metrics
            if len(test_actual) > 2 and np.std(test_actual) > 0:
                r, _ = pearsonr(test_predicted, test_actual)
                pearson_runs.append(r)

            rmse = float(np.sqrt(np.mean((test_actual - test_predicted) ** 2)))
            rmse_runs.append(rmse)

            # Violation rates
            residuals_cal = cal_actual - alpha0 * cal_pred_sums * np.power(
                np.maximum(cal_npar, 1.0), -beta)
            pos_res = residuals_cal[residuals_cal > 0]
            margin_95 = float(np.percentile(pos_res, 95)) if len(pos_res) > 0 else 0.0

            viol_raw = float(np.mean(test_actual > test_predicted))
            viol_95 = float(np.mean(test_actual > test_predicted + margin_95))
            viol_raw_runs.append(viol_raw)
            viol_95_runs.append(viol_95)

        if not pearson_runs:
            continue

        row = {
            "k": k,
            "pearson_mean": float(np.mean(pearson_runs)),
            "pearson_std": float(np.std(pearson_runs)),
            "rmse_mean": float(np.mean(rmse_runs)),
            "rmse_std": float(np.std(rmse_runs)),
            "viol_raw_mean": float(np.mean(viol_raw_runs)),
            "viol_raw_std": float(np.std(viol_raw_runs)),
            "viol_95_mean": float(np.mean(viol_95_runs)),
            "viol_95_std": float(np.std(viol_95_runs)),
            "n_repeats": len(pearson_runs),
        }
        results.append(row)
        print(f"{k:>5} {row['pearson_mean']:>9.3f}±{row['pearson_std']:.3f} "
              f"{row['rmse_mean']:>9.3f}±{row['rmse_std']:.3f} "
              f"{row['viol_raw_mean']:>9.1%}±{row['viol_raw_std']:.1%} "
              f"{row['viol_95_mean']:>9.1%}±{row['viol_95_std']:.1%}")

    output = {
        "graph_data": args.graph_data,
        "n_layers": n_layers,
        "n_total_masks": n_total,
        "n_singletons": len(singleton_idx),
        "n_repeats": args.n_repeats,
        "curve": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
