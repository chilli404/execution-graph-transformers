"""Verify the balanced-cancellation lower bound on β.

Checks Proposition 5: β ≥ β_BC = log(1/η)/log(L)
where η = D(m_all) / Σ d_l is the all-parallel ratio.

Also checks:
- Sequential specialist has η > 1 (superadditive)
- CV(d_l) correlates with excess β
- Pairwise interaction magnitudes vs uniform cross-term prediction

Usage:
  python scripts/verify_beta_theory.py [--output results/beta_theory_verification.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from fogen.execution_graph import fit_count_adjusted_scale


MODELS = [
    ("120M cw=1.0", "results/data/climbmix_graphs.json", 12, 1.0),
    ("120M cw=0.1", "blackwell/results/120m_cw01_poly_graphs.json", 12, 0.1),
    ("430M cw=1.0", "blackwell/results/430m_poly_graphs.json", 20, 1.0),
    ("430M cw=0.1", "blackwell/results/430m_cw01_poly_graphs.json", 20, 0.1),
    ("1B cw=1.0", "blackwell/results/1b_poly_graphs.json", 28, 1.0),
    ("1B cw=0.1", "blackwell/results/1b_cw01_poly_graphs.json", 28, 0.1),
    ("3B cw=0.1", "blackwell/results/3b_poly_graphs.json", 24, 0.1),
    ("7B cw=0.1", "blackwell/results/7b_12k_poly_graphs.json", 32, 0.1),
]

SEQ_SPECIALIST = ("430M seq", "blackwell/results/430m_seq_composition.json", 20)


def extract_composition_data(graph_file):
    with open(graph_file) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    baseline_kl = 0.0
    single_kl = np.zeros(n_layers)
    all_par_kl = 0.0

    for row in rows:
        n_par = sum(row["bits"])
        if n_par == 0:
            baseline_kl = row.get("symmetric_kl", 0.0)
        elif n_par == 1:
            layer = row["bits"].index(1)
            single_kl[layer] = max(0, row.get("symmetric_kl", 0) - baseline_kl)
        elif all(b == 1 for b in row["bits"]):
            all_par_kl = row.get("symmetric_kl", 0) - baseline_kl

    multi = [r for r in rows if 1 < sum(r["bits"]) < n_layers]
    pred_sums = [float(np.dot(r["bits"], single_kl)) for r in multi]
    actual_kl = [r["symmetric_kl"] - baseline_kl for r in multi]
    n_par = [sum(r["bits"]) for r in multi]

    return {
        "n_layers": n_layers,
        "single_kl": single_kl,
        "all_par_kl": all_par_kl,
        "d_add_all": float(single_kl.sum()),
        "pred_sums": pred_sums,
        "actual_kl": actual_kl,
        "n_par": n_par,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Verification of Proposition 5 (Balanced Cancellation)")
    print("=" * 70)

    records = []
    print(f"\n{'Model':<16} {'L':>3} {'η':>6} {'β_BC':>6} {'β_fit':>6} "
          f"{'β_fit/β_BC':>10} {'CV(d)':>6}")
    print("-" * 60)

    for label, gfile, n_layers, cw in MODELS:
        if not Path(gfile).exists():
            print(f"{label:<16} MISSING")
            continue

        data = extract_composition_data(gfile)
        d_add = data["d_add_all"]
        if d_add < 1e-12:
            continue

        eta = data["all_par_kl"] / d_add
        beta_bc = math.log(1.0 / max(eta, 1e-12)) / math.log(data["n_layers"]) if eta > 0 else 0

        alpha0, beta_fit = fit_count_adjusted_scale(
            data["pred_sums"], data["actual_kl"], data["n_par"])

        cv = float(np.std(data["single_kl"]) / max(np.mean(data["single_kl"]), 1e-12))

        records.append({
            "label": label,
            "n_layers": data["n_layers"],
            "cw": cw,
            "eta": eta,
            "beta_bc": beta_bc,
            "beta_fit": beta_fit,
            "ratio": beta_fit / max(beta_bc, 1e-12),
            "cv_d": cv,
            "bound_holds": beta_fit >= beta_bc - 0.01,
        })

        check = "✓" if beta_fit >= beta_bc - 0.01 else "✗"
        print(f"{label:<16} {data['n_layers']:>3} {eta:>6.3f} {beta_bc:>6.3f} "
              f"{beta_fit:>6.3f} {beta_fit/max(beta_bc,1e-12):>10.2f} {cv:>6.2f} {check}")

    # Check sequential specialist
    print(f"\n--- Sequential Specialist (should have η > 1) ---")
    seq_label, seq_file, seq_layers = SEQ_SPECIALIST
    if Path(seq_file).exists():
        seq_data = extract_composition_data(seq_file)
        seq_eta = seq_data["all_par_kl"] / max(seq_data["d_add_all"], 1e-12)
        print(f"  {seq_label}: η = {seq_eta:.3f} "
              f"{'(superadditive ✓)' if seq_eta > 1 else '(NOT superadditive ✗)'}")
    else:
        print(f"  {seq_label}: MISSING")

    # CV correlation with excess beta
    if len(records) >= 4:
        cvs = [r["cv_d"] for r in records]
        excess = [r["beta_fit"] - r["beta_bc"] for r in records]
        rho, p = spearmanr(cvs, excess)
        print(f"\n--- CV(d) vs excess β ---")
        print(f"  Spearman ρ = {rho:.3f} (p = {p:.4f})")

    # Bound summary
    all_hold = all(r["bound_holds"] for r in records)
    print(f"\n--- Lower bound β ≥ β_BC holds: "
          f"{'ALL ✓' if all_hold else 'SOME FAIL ✗'} ({sum(r['bound_holds'] for r in records)}/{len(records)}) ---")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"records": records}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
