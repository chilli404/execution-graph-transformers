"""Ternary composition predictor: predict quality cost from skip structure.

Tests whether per-layer skip costs predict multi-skip degradation,
analogous to the binary composition law for seq/par masks.

Usage:
  python scripts/eval_ternary_composition.py \
      --mask_eval paper_results/ternary/ternary_consistent_430m_mask_eval.json \
      --skip_costs paper_results/mechanism/layer_skip_430m_cw10.json \
      --output paper_results/ternary/ternary_composition.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as _minimize
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_eval", required=True)
    parser.add_argument("--skip_costs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.mask_eval) as f:
        eval_data = json.load(f)
    with open(args.skip_costs) as f:
        skip_data = json.load(f)

    n_layers = eval_data["n_layers"]
    rows = eval_data["rows"]

    # Per-layer skip costs from the single-layer-skip eval
    skip_costs = np.zeros(n_layers)
    for entry in skip_data["single_layer_skip"]:
        skip_costs[entry["layer"]] = max(0, entry["degradation"])

    print(f"=== Ternary Composition Predictor ({n_layers} layers, {len(rows)} masks) ===")
    print(f"Skip costs: min={skip_costs.min():.4f} max={skip_costs.max():.4f} mean={skip_costs.mean():.4f}")
    print()

    # For each mask, compute: sum of skip costs for skipped layers
    pred_additive = []
    actual = []
    n_skips = []
    for r in rows:
        mask = r["mask"]
        skip_sum = sum(skip_costs[i] for i, m in enumerate(mask) if m == "skip")
        pred_additive.append(skip_sum)
        actual.append(r["ternary_degradation"])
        n_skips.append(r["n_skip"])

    pred_additive = np.array(pred_additive)
    actual = np.array(actual)
    n_skips = np.array(n_skips)

    # Filter to masks with at least 1 skip
    has_skip = n_skips > 0
    pa = pred_additive[has_skip]
    ac = actual[has_skip]
    ns = n_skips[has_skip]

    # --- Model 1: Linear in skip count ---
    # ΔBPB ≈ c * n_skip
    c_linear = float(np.dot(ns, ac) / np.dot(ns, ns))
    pred_linear = c_linear * ns

    # --- Model 2: Additive skip costs ---
    # ΔBPB ≈ α * Σ skip_cost_l
    alpha_add = float(np.dot(pa, ac) / np.dot(pa, pa)) if np.dot(pa, pa) > 0 else 1.0
    pred_add = alpha_add * pa

    # --- Model 3: Count-adjusted (like binary) ---
    # ΔBPB ≈ α₀ * Σ skip_cost_l * n_skip^(-β)
    def ca_loss(params):
        a, b = params
        pred = a * pa * np.power(np.maximum(ns, 1.0).astype(float), -b)
        return float(np.mean((ac - pred) ** 2))

    best = None
    for a0 in [0.5, 1.0, 1.5, 2.0]:
        for b0 in [0.0, 0.1, 0.2, 0.3, 0.5]:
            res = _minimize(ca_loss, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best is None or res.fun < best.fun:
                best = res
    alpha_ca, beta_ca = float(best.x[0]), float(best.x[1])
    pred_ca = alpha_ca * pa * np.power(np.maximum(ns, 1.0).astype(float), -beta_ca)

    # --- 50/50 holdout ---
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(pa))
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    # Fit CA on train, evaluate on test
    pa_tr, ac_tr, ns_tr = pa[train_idx], ac[train_idx], ns[train_idx]
    pa_te, ac_te, ns_te = pa[test_idx], ac[test_idx], ns[test_idx]

    def ca_loss_tr(params):
        a, b = params
        pred = a * pa_tr * np.power(np.maximum(ns_tr, 1.0).astype(float), -b)
        return float(np.mean((ac_tr - pred) ** 2))

    best_tr = None
    for a0 in [0.5, 1.0, 1.5, 2.0]:
        for b0 in [0.0, 0.1, 0.2, 0.3, 0.5]:
            res = _minimize(ca_loss_tr, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best_tr is None or res.fun < best_tr.fun:
                best_tr = res
    a_ho, b_ho = float(best_tr.x[0]), float(best_tr.x[1])
    pred_ho = a_ho * pa_te * np.power(np.maximum(ns_te, 1.0).astype(float), -b_ho)

    def metrics(y_true, y_pred, label):
        r_p = pearsonr(y_true, y_pred).statistic if len(y_true) > 2 else 0
        r_s = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else 0
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        print(f"  {label:<35} Pearson={r_p:.4f}  Spearman={r_s:.4f}  RMSE={rmse:.4f}")
        return {"pearson": round(r_p, 4), "spearman": round(r_s, 4), "rmse": round(rmse, 4)}

    print("--- Full-data fits ---")
    m1 = metrics(ac, pred_linear, f"Linear: {c_linear:.4f}*n_skip")
    m2 = metrics(ac, pred_add, f"Additive: {alpha_add:.3f}*Σ skip_cost")
    m3 = metrics(ac, pred_ca, f"CA: {alpha_ca:.3f}*Σ skip_cost*n^(-{beta_ca:.3f})")
    print()
    print("--- Holdout (fit on 50%, test on 50%) ---")
    m_ho = metrics(ac_te, pred_ho, f"CA holdout: α={a_ho:.3f} β={b_ho:.3f}")

    result = {
        "n_layers": n_layers,
        "n_masks": len(rows),
        "n_masks_with_skip": int(has_skip.sum()),
        "skip_costs": skip_costs.tolist(),
        "models": {
            "linear": {"c": round(c_linear, 4), **m1},
            "additive": {"alpha": round(alpha_add, 4), **m2},
            "count_adjusted": {"alpha": round(alpha_ca, 4), "beta": round(beta_ca, 4), **m3},
        },
        "holdout": {
            "alpha": round(a_ho, 4), "beta": round(b_ho, 4),
            "n_train": half, "n_test": len(idx) - half,
            **m_ho,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
