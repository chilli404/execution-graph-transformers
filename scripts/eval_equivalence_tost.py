"""TOST equivalence test for downstream benchmark parity across execution graphs.

Performs Two One-Sided Tests (TOST) to establish that parallel and compiled
execution produce equivalent downstream accuracy to sequential execution,
within a pre-specified margin δ.

Uses aggregate statistics (mean, stderr, n) from lm-eval results.
With n > 1000 per task, the z-approximation is adequate.

Usage:
  python scripts/eval_equivalence_tost.py \
      --seq_dir blackwell/results/lm_eval_7b_seq \
      --alt_dirs blackwell/results/lm_eval_7b_par blackwell/results/lm_eval_7b_compiled \
      --delta 0.02 \
      [--output results/equivalence_tost.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from scipy.stats import norm


def load_results(result_dir: str) -> dict:
    """Load lm-eval results JSON from a directory."""
    files = glob.glob(f"{result_dir}/**/results_*.json", recursive=True)
    if not files:
        raise FileNotFoundError(f"No results JSON in {result_dir}")
    with open(files[0]) as f:
        return json.load(f)


def tost_z(mean1, se1, n1, mean2, se2, n2, delta):
    """Two One-Sided Tests using z-approximation for two independent proportions.

    Tests H0: |μ1 - μ2| ≥ δ  vs  H1: |μ1 - μ2| < δ

    Returns dict with test statistics, p-values, CI, and equivalence decision.
    """
    diff = mean2 - mean1
    se_diff = math.sqrt(se1**2 + se2**2)

    if se_diff < 1e-12:
        return {
            "diff": diff, "se_diff": 0, "equivalent": abs(diff) < delta,
            "z_upper": float("inf"), "z_lower": float("inf"),
            "p_upper": 0.0, "p_lower": 0.0,
            "ci90_lo": diff, "ci90_hi": diff,
        }

    z_upper = (diff + delta) / se_diff
    z_lower = (-diff + delta) / se_diff

    p_upper = 1 - norm.cdf(z_upper)
    p_lower = 1 - norm.cdf(z_lower)
    p_tost = max(p_upper, p_lower)

    ci90_lo = diff - norm.ppf(0.95) * se_diff
    ci90_hi = diff + norm.ppf(0.95) * se_diff

    return {
        "diff": diff,
        "se_diff": se_diff,
        "z_upper": z_upper,
        "z_lower": z_lower,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_tost": p_tost,
        "ci90_lo": ci90_lo,
        "ci90_hi": ci90_hi,
        "equivalent": p_tost < 0.05,
        "ci_within_delta": ci90_lo > -delta and ci90_hi < delta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--alt_dirs", nargs="+", required=True)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seq = load_results(args.seq_dir)
    tasks = [t for t in seq["results"] if t not in ("_all",)]
    metrics = ["acc,none", "acc_norm,none"]

    print(f"Equivalence margin δ = {args.delta} ({args.delta*100:.0f} percentage points)")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    all_results = []

    for alt_dir in args.alt_dirs:
        alt_name = Path(alt_dir).name
        alt = load_results(alt_dir)
        print(f"=== Sequential vs {alt_name} ===")
        print(f"{'Task':<14} {'Metric':<10} {'Δ':>7} {'90% CI':>16} "
              f"{'p(TOST)':>8} {'Equiv?':>7}")
        print("-" * 68)

        for task in tasks:
            seq_r = seq["results"][task]
            alt_r = alt["results"][task]
            n = seq_r.get("sample_len", seq.get("n-samples", {}).get(task, {}).get("effective", 1000))

            for metric in metrics:
                if metric not in seq_r or metric not in alt_r:
                    continue

                se_key = metric.replace(",none", "_stderr,none")
                result = tost_z(
                    mean1=seq_r[metric],
                    se1=seq_r.get(se_key, 0.01),
                    n1=n,
                    mean2=alt_r[metric],
                    se2=alt_r.get(se_key, 0.01),
                    n2=n,
                    delta=args.delta,
                )
                result["task"] = task
                result["metric"] = metric.split(",")[0]
                result["alt"] = alt_name
                all_results.append(result)

                eq_str = "YES" if result["equivalent"] else "no"
                m = metric.split(",")[0]
                print(f"{task:<14} {m:<10} {result['diff']:>+7.4f} "
                      f"[{result['ci90_lo']:>+7.4f}, {result['ci90_hi']:>+7.4f}] "
                      f"{result['p_tost']:>8.4f} {eq_str:>7}")

        print()

    n_tests = len(all_results)
    n_equiv = sum(r["equivalent"] for r in all_results)
    n_ci = sum(r["ci_within_delta"] for r in all_results)
    print(f"Summary: {n_equiv}/{n_tests} tests establish equivalence at δ={args.delta} (α=0.05)")
    print(f"         {n_ci}/{n_tests} have 90% CI entirely within ±δ")

    if args.output:
        out = {
            "delta": args.delta,
            "seq_dir": args.seq_dir,
            "alt_dirs": args.alt_dirs,
            "tests": all_results,
            "n_equivalent": n_equiv,
            "n_tests": n_tests,
        }
        def _default(obj):
            if isinstance(obj, bool):
                return bool(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Convert bools in results
        for r in out["tests"]:
            for k, v in r.items():
                if isinstance(v, bool):
                    r[k] = int(v)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
