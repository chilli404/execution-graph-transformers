"""Test compiler calibration on independently held-out masks.

Splits multi-layer masks into calibration (50%) and evaluation (50%) sets,
fits the composition model + calibration margins on the calibration set,
and measures violation rates on the held-out set. Repeats with 10 random
splits and reports mean ± std.

Usage:
  python scripts/eval_compiler_holdout_coverage.py \
      --graph_data paper_results/core_scaling/430m_poly_graphs.json \
      --output paper_results/composition/compiler_holdout_coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fogen.execution_graph import (
    compile_graph_greedy,
    fit_count_adjusted_scale,
    single_layer_effects,
)


def run_one_split(rows, kl_effects, baseline_kl, n_layers, rng, cal_frac=0.5):
    multi = [r for r in rows if sum(r["bits"]) > 1]
    idx = rng.permutation(len(multi))
    n_cal = int(len(multi) * cal_frac)
    cal_idx, eval_idx = idx[:n_cal], idx[n_cal:]

    cal_rows = [multi[i] for i in cal_idx]
    eval_rows = [multi[i] for i in eval_idx]

    # Fit on calibration set
    cal_pred_sums = np.array([float(np.dot(r["bits"], kl_effects)) for r in cal_rows])
    cal_actual = np.array([r["symmetric_kl"] - baseline_kl for r in cal_rows])
    cal_n_par = np.array([sum(r["bits"]) for r in cal_rows], dtype=float)

    alpha0, beta = fit_count_adjusted_scale(cal_pred_sums, cal_actual, cal_n_par)
    cal_predicted = alpha0 * cal_pred_sums * np.power(np.maximum(cal_n_par, 1.0), -beta)
    cal_residuals = cal_actual - cal_predicted
    pos_resid = cal_residuals[cal_residuals > 0]
    if len(pos_resid) == 0:
        pos_resid = np.array([0.0])
    margin_95 = float(np.percentile(pos_resid, 95))
    margin_99 = float(np.percentile(pos_resid, 99))

    # Evaluate on held-out set
    eval_pred_sums = np.array([float(np.dot(r["bits"], kl_effects)) for r in eval_rows])
    eval_actual = np.array([r["symmetric_kl"] - baseline_kl for r in eval_rows])
    eval_n_par = np.array([sum(r["bits"]) for r in eval_rows], dtype=float)
    eval_predicted = alpha0 * eval_pred_sums * np.power(np.maximum(eval_n_par, 1.0), -beta)

    n_eval = len(eval_rows)
    raw_violations = int(np.sum(eval_actual > eval_predicted))
    cal95_violations = int(np.sum(eval_actual > eval_predicted + margin_95))
    cal99_violations = int(np.sum(eval_actual > eval_predicted + margin_99))

    # Compiler oracle comparison on held-out set
    # Find the mask with max parallel layers that stays within budget
    # Use median budget as a representative
    budgets = eval_predicted
    oracle_pars = eval_n_par
    compiler_results = []
    for budget_pct in [25, 50, 75]:
        budget = float(np.percentile(eval_actual[eval_actual > 0], budget_pct))
        compiled = compile_graph_greedy(
            kl_effects, np.ones(n_layers), budget, scale=alpha0)
        comp_bits = tuple(compiled["bits"])
        comp_n_par = sum(compiled["bits"])

        # Find oracle: mask in eval set with most parallel layers within budget
        valid = [(sum(r["bits"]), tuple(r["bits"]))
                 for r in eval_rows
                 if r["symmetric_kl"] - baseline_kl <= budget]
        oracle_n_par = max(v[0] for v in valid) if valid else 0

        compiler_results.append({
            "budget_percentile": budget_pct,
            "budget": budget,
            "compiler_n_parallel": comp_n_par,
            "oracle_n_parallel": oracle_n_par,
            "recovered_fraction": comp_n_par / max(oracle_n_par, 1),
        })

    return {
        "n_cal": n_cal,
        "n_eval": n_eval,
        "alpha0": alpha0,
        "beta": beta,
        "margin_95": margin_95,
        "margin_99": margin_99,
        "raw_violation_rate": raw_violations / n_eval,
        "cal95_violation_rate": cal95_violations / n_eval,
        "cal99_violation_rate": cal99_violations / n_eval,
        "compiler_comparison": compiler_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    kl_effects = single_layer_effects(rows, "symmetric_kl")
    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)

    rng = np.random.default_rng(args.seed)
    splits = []
    for i in range(args.n_splits):
        result = run_one_split(rows, kl_effects, baseline_kl, n_layers, rng)
        splits.append(result)
        print(f"Split {i+1}/{args.n_splits}: raw={result['raw_violation_rate']:.1%} "
              f"cal95={result['cal95_violation_rate']:.1%} "
              f"cal99={result['cal99_violation_rate']:.1%} "
              f"α₀={result['alpha0']:.3f} β={result['beta']:.3f}")

    # Aggregate
    raw_rates = [s["raw_violation_rate"] for s in splits]
    cal95_rates = [s["cal95_violation_rate"] for s in splits]
    cal99_rates = [s["cal99_violation_rate"] for s in splits]

    print(f"\n{'Metric':<25} {'Mean':>8} {'Std':>8}")
    print("-" * 45)
    print(f"{'Raw violation rate':<25} {np.mean(raw_rates):>8.1%} {np.std(raw_rates):>8.1%}")
    print(f"{'Calibrated 95% rate':<25} {np.mean(cal95_rates):>8.1%} {np.std(cal95_rates):>8.1%}")
    print(f"{'Calibrated 99% rate':<25} {np.mean(cal99_rates):>8.1%} {np.std(cal99_rates):>8.1%}")

    summary = {
        "graph_data": args.graph_data,
        "n_layers": n_layers,
        "n_splits": args.n_splits,
        "raw_violation": {"mean": float(np.mean(raw_rates)), "std": float(np.std(raw_rates))},
        "cal95_violation": {"mean": float(np.mean(cal95_rates)), "std": float(np.std(cal95_rates))},
        "cal99_violation": {"mean": float(np.mean(cal99_rates)), "std": float(np.std(cal99_rates))},
        "splits": splits,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
