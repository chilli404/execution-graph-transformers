"""Mechanistic analysis of why the composition law works.

Investigates three hypotheses for the |m|^(-β) subadditivity:
  H1: Residual stream dilution — perturbations shrink relative to
      a growing residual norm as more layers contribute.
  H2: Defect anticorrelation — per-layer defect vectors partially
      cancel when multiple layers are switched simultaneously.
  H3: The composition law is a consequence of the model's depth
      structure: deeper perturbations have less room to compound.

Loads per-layer defect data and multi-mask graph evaluations,
performs correlation/cancellation analysis, and tests alternative
functional forms against |m|^(-β).

Usage:
  python scripts/analyze_composition_mechanism.py \
      --graph_data blackwell/results/430m_poly_graphs.json \
      [--plot composition_mechanism.pdf]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as _minimize
from scipy.stats import pearsonr, spearmanr


def fit_ca(pred_sums, actual, n_par):
    pred_sums = np.asarray(pred_sums, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n_par = np.asarray(n_par, dtype=float)

    def loss(params):
        a, b = params
        p = a * pred_sums * np.power(np.maximum(n_par, 1.0), -b)
        return float(np.mean((actual - p) ** 2))

    best = None
    for a0 in [0.8, 1.0, 1.2, 1.5]:
        for b0 in [0.1, 0.2, 0.3, 0.4]:
            res = _minimize(loss, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best is None or res.fun < best.fun:
                best = res
    return float(best.x[0]), float(best.x[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True, nargs="+",
                        help="One or more graph-rewrite JSON files")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    all_results = []

    for graph_file in args.graph_data:
        print(f"\n{'=' * 70}")
        print(f"Analyzing: {graph_file}")
        print("=" * 70)

        with open(graph_file) as f:
            data = json.load(f)

        rows = data["rows"]
        layer_defects = np.array(data["layer_defects"])
        n_layers = len(layer_defects)

        # Extract single-layer and multi-layer data
        baseline_kl = 0.0
        single_kl = np.zeros(n_layers)
        for row in rows:
            n_par = sum(row["bits"])
            if n_par == 0:
                baseline_kl = row["symmetric_kl"]
            elif n_par == 1:
                layer = row["bits"].index(1)
                single_kl[layer] = row["symmetric_kl"] - baseline_kl

        multi = [r for r in rows if sum(r["bits"]) > 1]
        bits_matrix = np.array([r["bits"] for r in multi])
        actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
        pred_sums = bits_matrix @ single_kl
        n_par = bits_matrix.sum(axis=1)

        # ============================================================
        # H1: Residual Stream Dilution
        # ============================================================
        # If each layer adds ~equal contribution to the residual,
        # the relative magnitude of a perturbation at layer l
        # shrinks as 1/(total layers contributing). With |m| layers
        # perturbed, the aggregate effect should scale sub-linearly.
        #
        # Test: does the ratio actual_kl / pred_sum correlate with
        # 1/|m|^β, and does β ≈ what we'd predict from dilution?
        print(f"\n--- H1: Residual Stream Dilution ---")
        alpha0, beta = fit_ca(pred_sums, actual_kl, n_par)
        print(f"Fitted: α₀={alpha0:.4f}, β={beta:.4f}")

        ratio = np.where(pred_sums > 0, actual_kl / pred_sums, np.nan)
        valid = ~np.isnan(ratio)
        if valid.sum() > 2:
            rho_ratio_n, _ = spearmanr(n_par[valid], ratio[valid])
            print(f"Spearman(|m|, actual/predicted): {rho_ratio_n:.4f}")
            print(f"  (negative = subadditive, i.e. more parallel layers -> lower ratio)")

        # Group by |m| and show mean ratio
        print(f"\n  |m|  mean(actual/Σd)  std    n")
        for k in sorted(set(n_par)):
            mask = n_par == k
            if mask.sum() == 0:
                continue
            r = ratio[mask & valid]
            if len(r) > 0:
                print(f"  {k:>3}  {r.mean():>14.4f}  {r.std():>6.4f}  {len(r):>3}")

        # ============================================================
        # H2: Defect Anticorrelation
        # ============================================================
        # If layer-level defect vectors are anticorrelated,
        # multi-layer perturbations partially cancel.
        # We can test this indirectly: for pairs of layers (i,j),
        # does KL(i+j) < KL(i) + KL(j)?
        print(f"\n--- H2: Pairwise Defect Interaction ---")
        pair_rows = [r for r in rows if sum(r["bits"]) == 2]
        if len(pair_rows) > 0:
            interactions = []
            for row in pair_rows:
                i, j = [l for l, b in enumerate(row["bits"]) if b]
                kl_pair = row["symmetric_kl"] - baseline_kl
                kl_sum = single_kl[i] + single_kl[j]
                interaction = kl_pair - kl_sum  # negative = cancellation
                interactions.append({
                    "layers": (i, j),
                    "kl_pair": kl_pair,
                    "kl_additive": kl_sum,
                    "interaction": interaction,
                    "distance": abs(i - j),
                })

            int_vals = [x["interaction"] for x in interactions]
            distances = [x["distance"] for x in interactions]
            n_negative = sum(1 for v in int_vals if v < 0)
            print(f"Pairs evaluated: {len(interactions)}")
            print(f"Subadditive (KL_pair < KL_i + KL_j): {n_negative}/{len(interactions)} "
                  f"({100*n_negative/len(interactions):.0f}%)")
            print(f"Mean interaction: {np.mean(int_vals):.4f} "
                  f"(negative = cancellation)")
            print(f"Median interaction: {np.median(int_vals):.4f}")

            if len(distances) > 2:
                rho_dist, p_dist = spearmanr(distances, int_vals)
                print(f"Spearman(layer_distance, interaction): {rho_dist:.4f} (p={p_dist:.4f})")
                print(f"  (negative = distant layers cancel more)")
        else:
            print("No pair-of-layer masks available.")

        # ============================================================
        # H3: Depth Structure — do later layers contribute less?
        # ============================================================
        print(f"\n--- H3: Depth Structure ---")
        print(f"Layer defects (L2 norm of seq→par difference):")
        for l in range(n_layers):
            bar = "█" * int(layer_defects[l] * 50 / max(layer_defects.max(), 1e-9))
            print(f"  L{l:>2}: {layer_defects[l]:.4f}  {bar}")

        rho_depth, _ = spearmanr(range(n_layers), layer_defects)
        print(f"Spearman(layer_index, defect): {rho_depth:.4f}")

        rho_depth_kl, _ = spearmanr(range(n_layers), single_kl)
        print(f"Spearman(layer_index, single-layer KL): {rho_depth_kl:.4f}")

        # ============================================================
        # Alternative functional forms
        # ============================================================
        print(f"\n--- Alternative Functional Forms ---")

        # Form 1: Linear (no subadditivity correction)
        def fit_linear(pred_sums, actual):
            alpha = float(np.dot(pred_sums, actual) / np.dot(pred_sums, pred_sums))
            pred = alpha * pred_sums
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            r = pearsonr(pred, actual).statistic if len(actual) > 2 else 0
            return {"name": "Linear: α·Σd", "params": 1, "rmse": rmse,
                    "pearson": r, "alpha": alpha}

        # Form 2: Count-adjusted (the paper's model)
        def fit_count_adj(pred_sums, actual, n_par):
            a, b = fit_ca(pred_sums, actual, n_par)
            pred = a * pred_sums * np.power(np.maximum(n_par, 1.0), -b)
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            r = pearsonr(pred, actual).statistic if len(actual) > 2 else 0
            return {"name": "CA: α₀·Σd·|m|^(-β)", "params": 2, "rmse": rmse,
                    "pearson": r, "alpha0": a, "beta": b}

        # Form 3: Sqrt scaling — D ∝ Σd / √|m|
        def fit_sqrt(pred_sums, actual, n_par):
            adj = pred_sums / np.sqrt(np.maximum(n_par, 1.0))
            alpha = float(np.dot(adj, actual) / np.dot(adj, adj))
            pred = alpha * adj
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            r = pearsonr(pred, actual).statistic if len(actual) > 2 else 0
            return {"name": "Sqrt: α·Σd/√|m|", "params": 1, "rmse": rmse,
                    "pearson": r}

        # Form 4: Log scaling — D ∝ Σd / log(|m|+1)
        def fit_log(pred_sums, actual, n_par):
            adj = pred_sums / np.log(np.maximum(n_par, 1.0) + 1)
            alpha = float(np.dot(adj, actual) / np.dot(adj, adj))
            pred = alpha * adj
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            r = pearsonr(pred, actual).statistic if len(actual) > 2 else 0
            return {"name": "Log: α·Σd/log(|m|+1)", "params": 1, "rmse": rmse,
                    "pearson": r}

        # Form 5: 3-parameter (overfit test) — α₀·Σd^γ·|m|^(-β)
        def fit_3p(pred_sums, actual, n_par):
            def loss(params):
                a, b, g = params
                p = a * np.power(np.maximum(pred_sums, 1e-12), g) * \
                    np.power(np.maximum(n_par, 1.0), -b)
                return float(np.mean((actual - p) ** 2))
            best = None
            for a0 in [0.8, 1.2]:
                for b0 in [0.1, 0.3]:
                    for g0 in [0.8, 1.0, 1.2]:
                        res = _minimize(loss, [a0, b0, g0], method="Nelder-Mead",
                                        options={"maxiter": 20000})
                        if best is None or res.fun < best.fun:
                            best = res
            a, b, g = best.x
            pred = a * np.power(np.maximum(pred_sums, 1e-12), g) * \
                np.power(np.maximum(n_par, 1.0), -b)
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            r = pearsonr(pred, actual).statistic if len(actual) > 2 else 0
            return {"name": f"3p: α·(Σd)^{g:.2f}·|m|^(-β)", "params": 3,
                    "rmse": rmse, "pearson": r, "gamma": g, "beta": b}

        forms = [
            fit_linear(pred_sums, actual_kl),
            fit_count_adj(pred_sums, actual_kl, n_par),
            fit_sqrt(pred_sums, actual_kl, n_par),
            fit_log(pred_sums, actual_kl, n_par),
            fit_3p(pred_sums, actual_kl, n_par),
        ]

        print(f"{'Form':<30} {'p':>2} {'RMSE':>8} {'Pearson':>8} {'ΔRMSE vs CA':>12}")
        print("-" * 65)
        ca_rmse = forms[1]["rmse"]
        for f in forms:
            delta = f["rmse"] - ca_rmse
            print(f"{f['name']:<30} {f['params']:>2} {f['rmse']:>8.4f} "
                  f"{f['pearson']:>8.4f} {delta:>+12.4f}")

        print(f"\n  Key insight: if 3p γ ≈ 1.0, then the Σd term is already linear")
        print(f"  and the 2p model is the right order. 3p γ = {forms[4].get('gamma', '?'):.3f}")

        result = {
            "file": graph_file,
            "n_layers": n_layers,
            "beta": alpha0,
            "forms": {f["name"]: {"rmse": f["rmse"], "pearson": f["pearson"]}
                      for f in forms},
        }
        all_results.append(result)

    if args.plot and len(args.graph_data) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use the first file for the detailed plot
        with open(args.graph_data[0]) as f:
            data = json.load(f)
        rows = data["rows"]
        layer_defects = np.array(data["layer_defects"])
        n_layers = len(layer_defects)

        baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
        single_kl = np.zeros(n_layers)
        for r in rows:
            if sum(r["bits"]) == 1:
                single_kl[r["bits"].index(1)] = r["symmetric_kl"] - baseline_kl

        multi = [r for r in rows if sum(r["bits"]) > 1]
        n_par = np.array([sum(r["bits"]) for r in multi])
        actual_kl = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
        pred_sums = np.array([sum(b * d for b, d in zip(r["bits"], single_kl)) for r in multi])

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel 1: Ratio actual/predicted vs |m|
        ax = axes[0]
        ratio = np.where(pred_sums > 0, actual_kl / pred_sums, np.nan)
        valid = ~np.isnan(ratio)
        ax.scatter(n_par[valid], ratio[valid], alpha=0.3, s=15)
        # Fit curve
        unique_m = sorted(set(n_par[valid]))
        mean_ratios = [ratio[valid & (n_par == m)].mean() for m in unique_m]
        ax.plot(unique_m, mean_ratios, "r-o", markersize=4, label="mean")
        a0, b0 = fit_ca(pred_sums[valid], actual_kl[valid], n_par[valid])
        m_range = np.linspace(1, max(unique_m), 100)
        ax.plot(m_range, a0 * m_range ** (-b0), "k--",
                label=f"α₀·|m|^(-{b0:.2f})", alpha=0.7)
        ax.set_xlabel("|m| (parallel layers)")
        ax.set_ylabel("actual KL / Σ single-layer KL")
        ax.legend(fontsize=8)
        ax.set_title("Subadditivity ratio")

        # Panel 2: Layer defect profile
        ax = axes[1]
        ax.barh(range(n_layers), layer_defects, color="steelblue")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
        ax.set_xlabel("Layer defect (L2)")
        ax.set_title("Per-layer defect profile")
        ax.invert_yaxis()

        # Panel 3: Pairwise interaction vs distance
        ax = axes[2]
        pair_rows = [r for r in rows if sum(r["bits"]) == 2]
        if pair_rows:
            dists, ints = [], []
            for r in pair_rows:
                i, j = [l for l, b in enumerate(r["bits"]) if b]
                kl_pair = r["symmetric_kl"] - baseline_kl
                interaction = kl_pair - (single_kl[i] + single_kl[j])
                dists.append(abs(i - j))
                ints.append(interaction)
            ax.scatter(dists, ints, alpha=0.4, s=15)
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
            ax.set_xlabel("Layer distance |i-j|")
            ax.set_ylabel("Interaction (neg = cancellation)")
            ax.set_title("Pairwise defect interaction")

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
