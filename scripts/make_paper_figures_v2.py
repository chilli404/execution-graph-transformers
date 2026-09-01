"""Generate all paper figures from the complete result set.

Covers:
  Fig 1: Scaling — agreement, KL, BPB degradation across 120M–7B
  Fig 2: Specialist collapse vs polymorphic (430M)
  Fig 3: Consistency ablation (120M baselines)
  Fig 4: Composition law — predicted vs actual at 430M with CA fit
  Fig 5: Composition holdout — 10k masks, all splits
  Fig 6: Cross-sequence-length transfer
  Fig 7: β deconfounding — scale vs cw
  Fig 8: Hardware compilation — three platforms
  Fig 9: Downstream parity — lm-eval 7B
  Fig 10: vLLM serving — TPOT vs concurrency
  Fig 11: Mechanistic — subadditivity ratio and defect profiles

Usage:
  python scripts/make_paper_figures_v2.py
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import minimize as _minimize

ROOT = Path(__file__).resolve().parents[1]
BDATA = ROOT / "blackwell" / "results"
DATA = ROOT / "results" / "data"
FIGS = ROOT / "results" / "figures" / "paper"

COLORS = {
    "seq": "#2166ac",
    "par": "#d6604d",
    "compiled": "#4dac26",
    "poly": "#762a83",
    "specialist": "#e08214",
    "cw10": "#2166ac",
    "cw01": "#d6604d",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def fit_ca(pred_sums, actual, n_par):
    pred_sums = np.asarray(pred_sums, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n_par = np.asarray(n_par, dtype=float)
    def loss(p):
        pred = p[0] * pred_sums * np.power(np.maximum(n_par, 1.0), -p[1])
        return float(np.mean((actual - pred) ** 2))
    best = None
    for a0 in [0.8, 1.0, 1.2, 1.5]:
        for b0 in [0.1, 0.2, 0.3, 0.4]:
            res = _minimize(loss, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best is None or res.fun < best.fun:
                best = res
    return float(best.x[0]), float(best.x[1])


def extract_graph_stats(path):
    data = load(path)
    rows = data["rows"]
    seq = next(r for r in rows if sum(r["bits"]) == 0)
    par = next((r for r in rows if all(b == 1 for b in r["bits"])), None)
    return {
        "seq_bpb": seq["val_bpb"],
        "par_bpb": par["val_bpb"] if par else None,
        "agreement": par["argmax_agreement"] if par else None,
        "kl": par["symmetric_kl"] if par else None,
        "bpb_deg": par["bpb_degradation"] if par else None,
        "layer_defects": np.array(data["layer_defects"]),
        "rows": rows,
    }


# ================================================================
# Fig 1: Scaling — core metrics across model sizes
# ================================================================
def fig_scaling():
    scales = [
        ("120M", DATA / "climbmix_graphs.json", 120),
        ("430M", BDATA / "430m_poly_graphs.json", 430),
        ("1B", BDATA / "1b_poly_graphs.json", 1000),
        ("7B", BDATA / "7b_12k_poly_graphs.json", 7000),
    ]
    # Use the cw=1.0 versions where available
    scale_data = []
    for label, path, size in scales:
        if not path.exists():
            continue
        s = extract_graph_stats(path)
        scale_data.append((label, size, s))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    labels = [d[0] for d in scale_data]
    x = np.arange(len(labels))

    # Agreement
    ax = axes[0]
    vals = [d[2]["agreement"] * 100 for d in scale_data]
    bars = ax.bar(x, vals, color=COLORS["poly"], alpha=0.85)
    ax.set_ylabel("Top-1 Agreement (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(90, 100)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}%",
                ha="center", fontsize=8)
    ax.set_title("(a) Cross-graph agreement")

    # KL
    ax = axes[1]
    vals = [d[2]["kl"] for d in scale_data]
    bars = ax.bar(x, vals, color=COLORS["par"], alpha=0.85)
    ax.set_ylabel("Symmetric KL (nats/token)")
    ax.set_xticks(x, labels)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_title("(b) All-parallel KL divergence")

    # BPB degradation
    ax = axes[2]
    vals = [d[2]["bpb_deg"] for d in scale_data]
    bars = ax.bar(x, vals, color=COLORS["compiled"], alpha=0.85)
    ax.set_ylabel("BPB degradation")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.00005, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.set_title("(c) Max BPB degradation")

    fig.suptitle("Graph equivalence across model scale", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig1_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 2: Specialist collapse vs polymorphic
# ================================================================
def fig_specialist_collapse():
    poly = extract_graph_stats(BDATA / "430m_poly_graphs.json")
    spec = extract_graph_stats(BDATA / "430m_seq_composition.json")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    categories = ["Sequential\nSpecialist", "Polymorphic\n(cw=1.0)"]
    x = np.arange(2)

    # Agreement
    ax = axes[0]
    vals = [spec["agreement"] * 100, poly["agreement"] * 100]
    bars = ax.bar(x, vals, color=[COLORS["specialist"], COLORS["poly"]], alpha=0.85)
    ax.set_ylabel("Agreement under parallel (%)")
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("(a) Top-1 agreement")

    # KL (log scale)
    ax = axes[1]
    vals = [spec["kl"], poly["kl"]]
    bars = ax.bar(x, vals, color=[COLORS["specialist"], COLORS["poly"]], alpha=0.85)
    ax.set_ylabel("Symmetric KL (nats/token)")
    ax.set_yscale("log")
    ax.set_xticks(x, categories)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.3, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("(b) KL divergence")

    # BPB degradation
    ax = axes[2]
    vals = [spec["bpb_deg"], poly["bpb_deg"]]
    bars = ax.bar(x, vals, color=[COLORS["specialist"], COLORS["poly"]], alpha=0.85)
    ax.set_ylabel("BPB degradation")
    ax.set_xticks(x, categories)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("(c) BPB degradation")

    fig.suptitle("430M: Sequential specialist vs polymorphic under graph switching",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_specialist_collapse.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig2_specialist_collapse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 3: Consistency ablation
# ================================================================
def fig_ablation():
    methods = [
        ("Random\nmask 1×", DATA / "random_mask_execution.json"),
        ("Random\nmask 2×", DATA / "random_mask_compute_matched_execution.json"),
        ("Joint CE", DATA / "joint_ce_execution.json"),
        ("Consistency\n(ours)", DATA / "climbmix_execution.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    labels, agreements, kls = [], [], []
    for label, path in methods:
        d = load(path)
        labels.append(label)
        agreements.append(d["argmax_agreement"] * 100)
        kls.append(d["symmetric_kl"])

    x = np.arange(len(labels))
    colors = ["#bdbdbd", "#969696", "#636363", COLORS["poly"]]

    ax = axes[0]
    bars = ax.bar(x, agreements, color=colors, alpha=0.9)
    ax.set_ylabel("Cross-graph agreement (%)")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(65, 95)
    for bar, v in zip(bars, agreements):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                ha="center", fontsize=8)
    ax.set_title("(a) Agreement")

    ax = axes[1]
    bars = ax.bar(x, kls, color=colors, alpha=0.9)
    ax.set_ylabel("Symmetric KL (nats/token)")
    ax.set_xticks(x, labels, fontsize=8)
    for bar, v in zip(bars, kls):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.1f}",
                ha="center", fontsize=8)
    ax.set_title("(b) KL divergence")

    fig.suptitle("Training objective ablation (120M, ClimbMix)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig3_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 4: Composition law — scatter with CA fit at multiple scales
# ================================================================
def fig_composition():
    datasets = [
        ("430M (cw=1.0)", BDATA / "430m_poly_graphs.json", "tab:blue"),
        ("1B (cw=1.0)", BDATA / "1b_poly_graphs.json", "tab:orange"),
        ("7B (cw=0.1)", BDATA / "7b_12k_poly_graphs.json", "tab:green"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (label, path, color) in zip(axes, datasets):
        data = load(path)
        rows = data["rows"]
        n_layers = len(rows[0]["bits"])
        baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
        single_kl = np.zeros(n_layers)
        for r in rows:
            if sum(r["bits"]) == 1:
                single_kl[r["bits"].index(1)] = r["symmetric_kl"] - baseline_kl

        multi = [r for r in rows if sum(r["bits"]) > 1]
        pred = np.array([sum(b * d for b, d in zip(r["bits"], single_kl)) for r in multi])
        actual = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
        n_par = np.array([sum(r["bits"]) for r in multi])

        alpha0, beta = fit_ca(pred, actual, n_par)
        ca_pred = alpha0 * pred * np.power(np.maximum(n_par, 1.0), -beta)

        ax.scatter(ca_pred, actual, alpha=0.4, s=12, c=color)
        lim = max(ca_pred.max(), actual.max()) * 1.08
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=1)
        rho = np.corrcoef(ca_pred, actual)[0, 1]
        ax.set_xlabel("Predicted KL (CA model)")
        ax.set_ylabel("Observed KL")
        ax.set_title(f"{label}\nα₀={alpha0:.2f}, β={beta:.2f}, r={rho:.3f}")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

    fig.suptitle("Composition law: predicted vs observed KL", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_composition.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig4_composition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 5: Composition holdout — bar chart of Pearson across splits
# ================================================================
def fig_holdout():
    holdout_430 = load(BDATA / "430m_composition_holdout.json")
    holdout_120 = load(DATA / "climbmix_composition_holdout.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (holdout, title) in zip(axes, [
        (holdout_430, "430M (10,155 masks)"),
        (holdout_120, "120M (59 masks)"),
    ]):
        splits = holdout["splits"]
        names = []
        linear_r = []
        ca_r = []
        for name, vals in splits.items():
            if name == "5fold_cv":
                names.append("5-fold CV")
                linear_r.append(vals["avg_linear_pearson"])
                ca_r.append(vals["avg_ca_pearson"])
            else:
                display = {
                    "random_5050": "Random\n50/50",
                    "fit_leq10_pred_gt10": "Fit ≤L/2\npred >L/2",
                    "fit_leq6_pred_gt6": "Fit ≤L/2\npred >L/2",
                    "fit_gt10_pred_leq10": "Fit >L/2\npred ≤L/2",
                    "fit_gt6_pred_leq6": "Fit >L/2\npred ≤L/2",
                    "fit_even_pred_odd": "Even→\nOdd",
                }.get(name, name)
                names.append(display)
                linear_r.append(vals["linear_pearson"])
                ca_r.append(vals["ca_pearson"])

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, linear_r, w, label="Linear (1p)", color="#bdbdbd", alpha=0.9)
        ax.bar(x + w/2, ca_r, w, label="CA (2p)", color=COLORS["poly"], alpha=0.9)
        ax.set_xticks(x, names, fontsize=7)
        ax.set_ylabel("Held-out Pearson r")
        ax.set_ylim(0.82, 1.0)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(title)
        for i, (l, c) in enumerate(zip(linear_r, ca_r)):
            ax.text(i + w/2, c + 0.003, f"{c:.3f}", ha="center", fontsize=7)

    fig.suptitle("Composition law: held-out validation", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_holdout.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig5_holdout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 6: Cross-sequence-length transfer
# ================================================================
def fig_cross_seqlen():
    data = load(BDATA / "430m_cross_seqlen.json")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    seqlens = sorted(data["evaluations"].keys(), key=int)
    pearson_2p = [data["evaluations"][s]["transfer_2p_pearson"] for s in seqlens]
    rmse_2p = [data["evaluations"][s]["transfer_2p_rmse"] for s in seqlens]
    betas = [data["evaluations"][s]["local_beta"] for s in seqlens]

    ax = axes[0]
    ax.plot([int(s) for s in seqlens], pearson_2p, "o-", color=COLORS["poly"],
            markersize=7, linewidth=2)
    ax.set_xlabel("Test sequence length")
    ax.set_ylabel("Transfer Pearson r")
    ax.set_ylim(0.96, 0.98)
    ax.set_title("(a) Prediction accuracy")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([int(s) for s in seqlens])
    for s, r in zip(seqlens, pearson_2p):
        ax.annotate(f"{r:.3f}", (int(s), r), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")

    ax = axes[1]
    ax.plot([int(s) for s in seqlens], betas, "s-", color=COLORS["par"],
            markersize=7, linewidth=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Local β")
    ax.set_ylim(0.18, 0.23)
    ax.set_title("(b) β stability across seq lengths")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([int(s) for s in seqlens])
    ax.axhline(data["fit_beta"], color="gray", linestyle="--", alpha=0.5,
               label=f"Fit β={data['fit_beta']:.3f}")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Cross-sequence-length transfer (430M, fit at seqlen={data['fit_seqlen']})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_cross_seqlen.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig6_cross_seqlen.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 7: β deconfounding
# ================================================================
def fig_beta():
    models = [
        ("120M", 106.7e6, 1.0, DATA / "climbmix_graphs.json"),
        ("120M", 106.7e6, 0.1, BDATA / "120m_cw01_poly_graphs.json"),
        ("430M", 443.5e6, 1.0, BDATA / "430m_poly_graphs.json"),
        ("430M", 443.5e6, 0.1, BDATA / "430m_cw01_poly_graphs.json"),
        ("1B", 1082.1e6, 1.0, BDATA / "1b_poly_graphs.json"),
        ("1B", 1082.1e6, 0.1, BDATA / "1b_cw01_poly_graphs.json"),
        ("3B", 3674.2e6, 0.1, BDATA / "3b_poly_graphs.json"),
        ("7B", 8657.0e6, 0.1, BDATA / "7b_12k_poly_graphs.json"),
    ]

    records = []
    for label, n_params, cw, path in models:
        if not path.exists():
            continue
        data = load(path)
        rows = data["rows"]
        n_layers = len(rows[0]["bits"])
        baseline = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
        single = np.zeros(n_layers)
        for r in rows:
            if sum(r["bits"]) == 1:
                single[r["bits"].index(1)] = r["symmetric_kl"] - baseline
        multi = [r for r in rows if sum(r["bits"]) > 1]
        if len(multi) < 5:
            continue
        ps = np.array([sum(b * d for b, d in zip(r["bits"], single)) for r in multi])
        ak = np.array([r["symmetric_kl"] - baseline for r in multi])
        np_ = np.array([sum(r["bits"]) for r in multi])
        a0, beta = fit_ca(ps, ak, np_)
        records.append((label, n_params, cw, beta))

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, n_params, cw, beta in records:
        color = COLORS["cw10"] if cw == 1.0 else COLORS["cw01"]
        marker = "o" if cw == 1.0 else "s"
        ax.scatter(np.log10(n_params), beta, c=color, marker=marker, s=90, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(f"{label}", (np.log10(n_params), beta),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)

    # Legend
    ax.scatter([], [], c=COLORS["cw10"], marker="o", s=60, label="cw=1.0")
    ax.scatter([], [], c=COLORS["cw01"], marker="s", s=60, label="cw=0.1")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("log₁₀(parameters)")
    ax.set_ylabel("β (subadditivity exponent)")
    ax.set_title("Subadditivity increases with scale\n(β deconfounded from consistency weight)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_beta.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig7_beta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 8: Hardware compilation
# ================================================================
def fig_hardware():
    fig, ax = plt.subplots(figsize=(7, 3.8))

    hardware = ["L40S\n(CUDA)", "M4 Max\n(MPS)", "Blackwell\nMax-Q"]
    strategies = ["All-parallel\nfused", "Mixed\n7/12 par", "Workload-\ndependent"]
    speedups_lo = [1.20, 1.14, 1.00]
    speedups_hi = [1.54, 1.14, 1.09]

    x = np.arange(len(hardware))
    colors = ["#2166ac", "#4dac26", "#d6604d"]

    bars = ax.bar(x, speedups_hi, color=colors, alpha=0.85)
    ax.bar(x, speedups_lo, color=colors, alpha=0.4)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}\n{s}" for h, s in zip(hardware, strategies)], fontsize=8)
    ax.set_ylabel("Speedup over sequential")
    ax.set_ylim(0.95, 1.6)
    ax.set_title("Same checkpoint, hardware-specific compilation")

    for bar, lo, hi in zip(bars, speedups_lo, speedups_hi):
        if lo == hi:
            ax.text(bar.get_x() + bar.get_width()/2, hi + 0.01, f"{hi:.2f}×",
                    ha="center", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, hi + 0.01,
                    f"{lo:.2f}–{hi:.2f}×", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "fig8_hardware.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig8_hardware.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 9: Downstream parity (lm-eval 7B)
# ================================================================
def fig_downstream():
    modes = {}
    for mode_name in ["seq", "par", "compiled"]:
        pattern = str(BDATA / f"lm_eval_7b_{mode_name}" / "**" / "results_*.json")
        files = glob.glob(pattern, recursive=True)
        if files:
            modes[mode_name] = load(files[0])["results"]

    if len(modes) < 3:
        print("Skipping fig9: lm-eval results not found")
        return

    tasks = ["arc_easy", "hellaswag", "piqa", "winogrande"]
    task_labels = ["ARC-Easy", "HellaSwag", "PIQA", "WinoGrande"]
    metric = "acc,none"
    stderr_key = "acc_stderr,none"

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tasks))
    w = 0.25

    for i, (mode, label, color) in enumerate([
        ("seq", "Sequential", COLORS["seq"]),
        ("par", "All-parallel", COLORS["par"]),
        ("compiled", "Compiled", COLORS["compiled"]),
    ]):
        vals = [modes[mode][t][metric] for t in tasks]
        errs = [modes[mode][t][stderr_key] for t in tasks]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, label=label,
               color=color, alpha=0.85, capsize=3)

    ax.set_xticks(x, task_labels)
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("7B downstream benchmarks: all execution graphs equivalent")
    ax.set_ylim(0.2, 0.65)

    fig.tight_layout()
    fig.savefig(FIGS / "fig9_downstream.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig9_downstream.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 10: vLLM serving
# ================================================================
def fig_vllm():
    path = BDATA / "vllm_serving_430m_full.json"
    if not path.exists():
        print("Skipping fig10: vllm results not found")
        return

    data = load(path)
    rows = data.get("rows", data.get("results", []))
    if not rows:
        print("Skipping fig10: no rows in vllm results")
        return
    concurrencies = sorted(set(r["max_concurrency"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for mode, color, label in [
        ("sequential", COLORS["seq"], "Sequential"),
        ("parallel_fused", COLORS["par"], "Parallel fused"),
    ]:
        tpots = []
        throughputs = []
        for c in concurrencies:
            matching = [r for r in rows if r["max_concurrency"] == c and r["mode"] == mode]
            if matching:
                tpots.append(matching[0]["mean_tpot"])
                throughputs.append(matching[0]["output_token_throughput"])
            else:
                tpots.append(None)
                throughputs.append(None)

        valid_c = [c for c, t in zip(concurrencies, tpots) if t is not None]
        valid_t = [t for t in tpots if t is not None]
        valid_th = [t for t in throughputs if t is not None]

        axes[0].plot(valid_c, valid_t, "o-", color=color, label=label, markersize=5)
        axes[1].plot(valid_c, valid_th, "o-", color=color, label=label, markersize=5)

    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("Mean TPOT (ms)")
    axes[0].set_title("(a) Token latency")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xscale("log", base=2)

    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Throughput (tok/s)")
    axes[1].set_title("(b) Throughput")
    axes[1].set_xscale("log", base=2)

    fig.suptitle("vLLM continuous batching (430M, Blackwell)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_vllm.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig10_vllm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Fig 11: Mechanistic — ratio decay + defect profiles
# ================================================================
def fig_mechanism():
    datasets = [
        ("430M", BDATA / "430m_poly_graphs.json", "tab:blue"),
        ("1B", BDATA / "1b_poly_graphs.json", "tab:orange"),
        ("7B", BDATA / "7b_12k_poly_graphs.json", "tab:green"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Ratio decay
    ax = axes[0]
    for label, path, color in datasets:
        data = load(path)
        rows = data["rows"]
        n_layers = len(rows[0]["bits"])
        baseline = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
        single = np.zeros(n_layers)
        for r in rows:
            if sum(r["bits"]) == 1:
                single[r["bits"].index(1)] = r["symmetric_kl"] - baseline
        multi = [r for r in rows if sum(r["bits"]) > 1]
        n_par = np.array([sum(r["bits"]) for r in multi])
        pred = np.array([sum(b * d for b, d in zip(r["bits"], single)) for r in multi])
        actual = np.array([r["symmetric_kl"] - baseline for r in multi])
        ratio = np.where(pred > 0, actual / pred, np.nan)

        unique_m = sorted(set(n_par))
        means = [np.nanmean(ratio[n_par == m]) for m in unique_m]
        ax.plot(unique_m, means, "o-", color=color, label=label, markersize=3,
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel("|m| (number of parallel layers)")
    ax.set_ylabel("Actual KL / Σ single-layer KL")
    ax.set_title("(a) Residual stream dilution")
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)

    # Panel 2: Defect profiles
    ax = axes[1]
    for label, path, color in datasets:
        data = load(path)
        defects = np.array(data["layer_defects"])
        normed = defects / defects.max()
        ax.plot(range(len(normed)), normed, "-", color=color, label=label,
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized defect")
    ax.set_title("(b) Per-layer defect profile")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Mechanism: perturbation dilution across depth", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig11_mechanism.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig11_mechanism.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures...")
    fig_scaling();         print("  Fig 1: Scaling")
    fig_specialist_collapse(); print("  Fig 2: Specialist collapse")
    fig_ablation();        print("  Fig 3: Ablation")
    fig_composition();     print("  Fig 4: Composition law")
    fig_holdout();         print("  Fig 5: Holdout validation")
    fig_cross_seqlen();    print("  Fig 6: Cross-seqlen")
    fig_beta();            print("  Fig 7: β deconfounding")
    fig_hardware();        print("  Fig 8: Hardware compilation")
    fig_downstream();      print("  Fig 9: Downstream parity")
    fig_vllm();            print("  Fig 10: vLLM serving")
    fig_mechanism();       print("  Fig 11: Mechanism")
    print(f"\nAll figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
