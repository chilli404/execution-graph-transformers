"""Deconfound β from scale and consistency weight.

Collects (N_params, cw, β) across all available graph-rewrite data,
fits β = a + b·log(N) + c·log(cw), and reports coefficients with
bootstrap uncertainty. Produces a table and optional plot.

Usage:
  python scripts/analyze_beta_deconfounding.py [--plot beta_deconfounding.pdf]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as _minimize
from scipy.stats import spearmanr


def estimate_params(n_layer, d_model, vocab_size=8192):
    """Rough non-embedding parameter count for the fogen architecture."""
    d_ff = 4 * d_model
    # Per block: attn (4 * d^2) + ffn (3 * d * d_ff) + norms + ve
    attn = 4 * d_model * d_model
    ffn = 3 * d_model * d_ff
    block = attn + ffn
    embed = 2 * vocab_size * d_model  # wte + lm_head (untied)
    return n_layer * block + embed


def fit_ca(predicted_sums, actual, n_parallel):
    """Fit D(m) ≈ α₀ · Σd · |m|^(-β), return (alpha0, beta)."""
    predicted_sums = np.asarray(predicted_sums, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n_parallel = np.asarray(n_parallel, dtype=float)

    def loss(params):
        a, b = params
        pred = a * predicted_sums * np.power(np.maximum(n_parallel, 1.0), -b)
        return float(np.mean((actual - pred) ** 2))

    best = None
    for a0 in [0.8, 1.0, 1.2, 1.5]:
        for b0 in [0.1, 0.2, 0.3, 0.4]:
            res = _minimize(loss, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best is None or res.fun < best.fun:
                best = res
    return float(best.x[0]), float(best.x[1])


def extract_beta(graph_file):
    """Extract β from a graph-rewrite JSON by fitting the CA model on KL data."""
    with open(graph_file) as f:
        data = json.load(f)
    rows = data["rows"]
    n_layers = len(rows[0]["bits"])

    baseline_kl = 0.0
    single_effects = np.zeros(n_layers)
    for row in rows:
        n_par = sum(row["bits"])
        if n_par == 0:
            baseline_kl = row.get("symmetric_kl", 0.0)
        elif n_par == 1:
            layer = row["bits"].index(1)
            single_effects[layer] = max(0, row.get("symmetric_kl", 0) - baseline_kl)

    multi = [r for r in rows if sum(r["bits"]) > 1]
    if len(multi) < 5:
        return None, None

    pred_sums = [float(np.dot(r["bits"], single_effects)) for r in multi]
    actual_kl = [r["symmetric_kl"] - baseline_kl for r in multi]
    n_par = [sum(r["bits"]) for r in multi]

    alpha0, beta = fit_ca(pred_sums, actual_kl, n_par)
    return alpha0, beta


MODELS = [
    # (label, graph_file, n_layer, d_model, cw)
    ("120M cw=1.0", "results/data/climbmix_graphs.json", 12, 704, 1.0),
    ("120M cw=0.1", "blackwell/results/120m_cw01_poly_graphs.json", 12, 704, 0.1),
    ("430M cw=1.0", "blackwell/results/430m_poly_graphs.json", 20, 1152, 1.0),
    ("430M cw=0.1", "blackwell/results/430m_cw01_poly_graphs.json", 20, 1152, 0.1),
    ("1B cw=1.0", "blackwell/results/1b_poly_graphs.json", 28, 1536, 1.0),
    ("1B cw=0.1", "blackwell/results/1b_cw01_poly_graphs.json", 28, 1536, 0.1),
    ("3B cw=0.1", "blackwell/results/3b_poly_graphs.json", 24, 3072, 0.1),
    ("7B cw=0.1 (12k)", "blackwell/results/7b_12k_poly_graphs.json", 32, 4096, 0.1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", default=None, help="Save plot to this path")
    args = parser.parse_args()

    print("=" * 70)
    print("β Deconfounding Analysis")
    print("=" * 70)

    records = []
    print(f"\n{'Model':<20} {'N_params':>12} {'cw':>6} {'α₀':>8} {'β':>8}")
    print("-" * 60)
    for label, gfile, n_layer, d_model, cw in MODELS:
        if not Path(gfile).exists():
            print(f"{label:<20} {'MISSING':>12}")
            continue
        alpha0, beta = extract_beta(gfile)
        if beta is None:
            print(f"{label:<20} {'TOO FEW':>12}")
            continue
        n_params = estimate_params(n_layer, d_model)
        records.append((label, n_params, cw, alpha0, beta))
        print(f"{label:<20} {n_params/1e6:>10.1f}M {cw:>6.1f} {alpha0:>8.3f} {beta:>8.4f}")

    if len(records) < 3:
        print("\nToo few data points for regression.")
        return

    # Fit β = a + b·log(N) + c·log(cw)
    labels = [r[0] for r in records]
    log_N = np.array([np.log(r[1]) for r in records])
    log_cw = np.array([np.log(r[2]) for r in records])
    betas = np.array([r[4] for r in records])

    X = np.column_stack([np.ones(len(records)), log_N, log_cw])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, betas, rcond=None)
    predicted = X @ coeffs
    rmse = np.sqrt(np.mean((betas - predicted) ** 2))
    r2 = 1 - np.sum((betas - predicted) ** 2) / np.sum((betas - betas.mean()) ** 2)

    print(f"\n{'Regression: β = a + b·log(N) + c·log(cw)':}")
    print(f"  a (intercept) = {coeffs[0]:.4f}")
    print(f"  b (log N)     = {coeffs[1]:.4f}")
    print(f"  c (log cw)    = {coeffs[2]:.4f}")
    print(f"  R²            = {r2:.4f}")
    print(f"  RMSE          = {rmse:.4f}")

    # Bootstrap uncertainty
    rng = np.random.default_rng(42)
    n_boot = 10000
    boot_coeffs = np.zeros((n_boot, 3))
    n = len(records)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_b = X[idx]
        y_b = betas[idx]
        try:
            c, _, _, _ = np.linalg.lstsq(X_b, y_b, rcond=None)
            boot_coeffs[i] = c
        except np.linalg.LinAlgError:
            boot_coeffs[i] = coeffs

    print(f"\n{'Bootstrap 95% CI (10k resamples)':}")
    for name, j in [("a (intercept)", 0), ("b (log N)", 1), ("c (log cw)", 2)]:
        lo, hi = np.percentile(boot_coeffs[:, j], [2.5, 97.5])
        print(f"  {name:<16} = {coeffs[j]:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    # Residual table
    print(f"\n{'Model':<20} {'β_actual':>8} {'β_pred':>8} {'residual':>8}")
    print("-" * 48)
    for i, (label, n_params, cw, alpha0, beta) in enumerate(records):
        print(f"{label:<20} {beta:>8.4f} {predicted[i]:>8.4f} {beta - predicted[i]:>+8.4f}")

    # Alternative: cw-only and scale-only regressions for comparison
    print(f"\n{'--- Ablation: scale-only model (β = a + b·log(N)) ---':}")
    X_scale = np.column_stack([np.ones(n), log_N])
    c_s, _, _, _ = np.linalg.lstsq(X_scale, betas, rcond=None)
    pred_s = X_scale @ c_s
    r2_s = 1 - np.sum((betas - pred_s) ** 2) / np.sum((betas - betas.mean()) ** 2)
    print(f"  R² = {r2_s:.4f} (vs {r2:.4f} for full model)")

    print(f"\n{'--- Ablation: cw-only model (β = a + c·log(cw)) ---':}")
    X_cw = np.column_stack([np.ones(n), log_cw])
    c_c, _, _, _ = np.linalg.lstsq(X_cw, betas, rcond=None)
    pred_c = X_cw @ c_c
    r2_c = 1 - np.sum((betas - pred_c) ** 2) / np.sum((betas - betas.mean()) ** 2)
    print(f"  R² = {r2_c:.4f} (vs {r2:.4f} for full model)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # Left: β vs log(N), colored by cw
        ax = axes[0]
        for label, n_params, cw, alpha0, beta in records:
            color = "tab:blue" if cw == 1.0 else "tab:orange"
            marker = "o" if cw == 1.0 else "s"
            ax.scatter(np.log10(n_params), beta, c=color, marker=marker,
                       s=80, zorder=3, label=f"cw={cw}" if label.startswith("120M") else "")
            ax.annotate(label.split()[0], (np.log10(n_params), beta),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("log₁₀(N parameters)")
        ax.set_ylabel("β (subadditivity)")
        ax.legend()
        ax.set_title("β vs scale, by consistency weight")

        # Right: actual vs predicted
        ax = axes[1]
        ax.scatter(predicted, betas, c="tab:green", s=80, zorder=3)
        lo = min(min(predicted), min(betas)) - 0.02
        hi = max(max(predicted), max(betas)) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
        for i, (label, _, _, _, beta) in enumerate(records):
            ax.annotate(label, (predicted[i], beta),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_xlabel("Predicted β")
        ax.set_ylabel("Actual β")
        ax.set_title(f"β regression (R²={r2:.3f})")

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
