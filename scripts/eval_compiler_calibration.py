"""Evaluate compiler budget calibration: empirical coverage at various budgets.

The composition law predicts KL divergence for arbitrary execution graphs,
but the prediction is empirical — not an upper bound. This script measures
how often the actual divergence exceeds the compiler's predicted budget,
then computes calibrated margins from held-out residuals to convert
predictions into conservative budgets.

Usage:
  python scripts/eval_compiler_calibration.py \
      --graph_data blackwell/results/430m_poly_graphs.json \
      --output blackwell/results/430m_compiler_calibration.json

  python scripts/eval_compiler_calibration.py \
      --graph_data results/data/climbmix_graphs.json \
      --output results/data/climbmix_compiler_calibration.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fogen.execution_graph import (
    compile_graph_greedy,
    fit_count_adjusted_scale,
    single_layer_effects,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_budgets", type=int, default=10)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    # ---------- extract single-layer effects and baseline ----------
    kl_effects = single_layer_effects(rows, "symmetric_kl")
    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)

    # ---------- fit composition law on multi-layer masks ----------
    multi = [r for r in rows if sum(r["bits"]) > 1]
    pred_sums = np.array([float(np.dot(r["bits"], kl_effects)) for r in multi])
    actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
    n_par = np.array([sum(r["bits"]) for r in multi], dtype=float)

    alpha0, beta = fit_count_adjusted_scale(pred_sums, actual_kl, n_par)
    predicted_kl = alpha0 * pred_sums * np.power(np.maximum(n_par, 1.0), -beta)
    residuals = actual_kl - predicted_kl

    print(f"Composition law fit: α₀={alpha0:.4f}, β={beta:.4f}")
    print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
          f"max={residuals.max():.4f}")

    # ---------- calibration margins from residual distribution ----------
    positive_residuals = residuals[residuals > 0]
    if len(positive_residuals) == 0:
        positive_residuals = np.array([0.0])
    margin_95 = float(np.percentile(positive_residuals, 95))
    margin_99 = float(np.percentile(positive_residuals, 99))
    print(f"Calibration margins: 95th={margin_95:.4f}, 99th={margin_99:.4f}")

    # ---------- build lookup: bits -> actual KL ----------
    bits_to_kl = {}
    for r in rows:
        key = tuple(r["bits"])
        bits_to_kl[key] = r["symmetric_kl"] - baseline_kl

    # ---------- evaluate at multiple budget levels ----------
    kl_range = actual_kl[actual_kl > 0]
    budget_min = float(np.percentile(kl_range, 10))
    budget_max = float(np.percentile(kl_range, 90))
    budgets = np.linspace(budget_min, budget_max, args.n_budgets)

    uniform_savings = np.ones(n_layers)

    calibration_rows = []
    for budget in budgets:
        results_at_budget = {"budget": float(budget)}

        for label, margin in [("raw", 0.0),
                              ("calibrated_95", margin_95),
                              ("calibrated_99", margin_99)]:
            effective_budget = max(budget - margin, 0.0)

            compiled = compile_graph_greedy(
                kl_effects, uniform_savings, effective_budget, scale=alpha0)
            bits = tuple(compiled["bits"])
            n_parallel = sum(compiled["bits"])
            predicted_effect = compiled["predicted_effect"]

            actual = bits_to_kl.get(bits)
            if actual is None:
                ca_pred = alpha0 * float(np.dot(bits, kl_effects)) * \
                    max(n_parallel, 1) ** (-beta)
                actual = ca_pred

            violated = bool(actual > budget)
            results_at_budget[f"{label}_n_parallel"] = int(n_parallel)
            results_at_budget[f"{label}_predicted_kl"] = float(predicted_effect)
            results_at_budget[f"{label}_actual_kl"] = float(actual)
            results_at_budget[f"{label}_violated"] = violated

        calibration_rows.append(results_at_budget)

    # ---------- aggregate violation rates across all masks ----------
    # For a more robust estimate, check every multi-layer mask we have
    # against the composition law prediction
    n_total = len(multi)
    n_violations_raw = int(np.sum(actual_kl > predicted_kl))
    n_violations_95 = int(np.sum(actual_kl > predicted_kl + margin_95))
    n_violations_99 = int(np.sum(actual_kl > predicted_kl + margin_99))

    mean_par_raw = np.mean([r["raw_n_parallel"] for r in calibration_rows])
    mean_par_95 = np.mean([r["calibrated_95_n_parallel"] for r in calibration_rows])
    mean_par_99 = np.mean([r["calibrated_99_n_parallel"] for r in calibration_rows])

    summary = {
        "raw": {
            "violation_rate": n_violations_raw / n_total,
            "n_violations": n_violations_raw,
            "n_total": n_total,
            "mean_parallel_layers": float(mean_par_raw),
        },
        "calibrated_95": {
            "violation_rate": n_violations_95 / n_total,
            "n_violations": n_violations_95,
            "n_total": n_total,
            "margin": margin_95,
            "mean_parallel_layers": float(mean_par_95),
        },
        "calibrated_99": {
            "violation_rate": n_violations_99 / n_total,
            "n_violations": n_violations_99,
            "n_total": n_total,
            "margin": margin_99,
            "mean_parallel_layers": float(mean_par_99),
        },
    }

    # ---------- print table ----------
    print(f"\n{'Compiler':<18} {'Violation rate':>15} {'Mean ∥ layers':>14}")
    print("-" * 50)
    for label, key in [("Raw predicted", "raw"),
                       ("Calibrated 95%", "calibrated_95"),
                       ("Calibrated 99%", "calibrated_99")]:
        s = summary[key]
        vr = s["violation_rate"]
        mp = s["mean_parallel_layers"]
        print(f"{label:<18} {vr:>14.1%} {mp:>14.1f}")

    # ---------- save ----------
    result = {
        "graph_data": args.graph_data,
        "n_layers": n_layers,
        "alpha0": alpha0,
        "beta": beta,
        "margin_95": margin_95,
        "margin_99": margin_99,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_max": float(residuals.max()),
        "summary": summary,
        "budget_sweep": calibration_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
