import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"


def load(name):
    with open(DATA / name) as handle:
        return json.load(handle)


def composition_points(data, field):
    rows = data["rows"]
    baseline = next(row[field] for row in rows if sum(row["bits"]) == 0)
    singles = {}
    for row in rows:
        if sum(row["bits"]) == 1:
            singles[row["bits"].index(1)] = row[field] - baseline
    predicted, actual = [], []
    for row in rows:
        if sum(row["bits"]) < 2:
            continue
        predicted.append(sum(
            singles[layer] for layer, bit in enumerate(row["bits"]) if bit))
        actual.append(row[field] - baseline)
    return np.asarray(predicted), np.asarray(actual)


def plot_composition():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for axis, (label, name) in zip(axes, [
        ("ClimbMix", "climbmix_graphs.json"),
        ("TinyStories", "tinystories_graphs.json"),
    ]):
        predicted, actual = composition_points(load(name), "symmetric_kl")
        slope = np.dot(predicted, actual) / np.dot(predicted, predicted)
        limit = max(predicted.max(), actual.max()) * 1.05
        axis.scatter(predicted, actual, alpha=0.75, s=22)
        axis.plot([0, limit], [0, slope * limit], color="black", linewidth=1.5)
        axis.set_title(f"{label} ($\\alpha$={slope:.2f})")
        axis.set_xlabel("Sum of single-layer KL effects")
        axis.set_ylabel("Observed multi-graph KL")
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
    fig.tight_layout()
    fig.savefig(FIGURES / "composition_law.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "composition_law.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def kv_speedups(data):
    groups = {}
    for row in data["rows"]:
        groups.setdefault(
            (row["batch_size"], row["prompt_length"]), {})[row["mode"]] = row
    ttft, tpot = [], []
    for modes in groups.values():
        ttft.append(
            modes["sequential"]["ttft_seconds"] /
            modes["parallel_fused"]["ttft_seconds"])
        tpot.append(
            modes["sequential"]["tpot_seconds"] /
            modes["parallel_fused"]["tpot_seconds"])
    return np.mean(ttft), np.mean(tpot)


def plot_serving():
    climb = kv_speedups(load("climbmix_kv.json"))
    tiny = kv_speedups(load("tinystories_kv.json"))
    values = np.asarray([climb, tiny])
    x = np.arange(2)
    width = 0.34
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.bar(x - width / 2, values[:, 0], width, label="TTFT")
    axis.bar(x + width / 2, values[:, 1], width, label="TPOT")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, ["ClimbMix", "TinyStories"])
    axis.set_ylabel("Fused-parallel speedup")
    axis.set_ylim(1.0, max(values.flatten()) * 1.12)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "serving_speedup.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "serving_speedup.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation():
    values = [
        load("random_mask_execution.json")["argmax_agreement"],
        load("random_mask_compute_matched_execution.json")["argmax_agreement"],
        load("joint_ce_execution.json")["argmax_agreement"],
        load("climbmix_execution.json")["argmax_agreement"],
    ]
    labels = ["Random\n1x compute", "Random\n2x compute", "Joint CE", "Consistency"]
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    bars = axis.bar(labels, values)
    axis.set_ylabel("Cross-graph argmax agreement")
    axis.set_ylim(0.7, 0.95)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value + 0.006,
            f"{100 * value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "consistency_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "consistency_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prompt_compiler():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for axis, (label, name) in zip(axes, [
        ("Within ClimbMix", "prompt_compiler_within.json"),
        ("ClimbMix to TinyStories", "prompt_compiler_cross.json"),
    ]):
        summaries = list(load(name)["summaries"].values())
        x = np.arange(len(summaries))
        width = 0.34
        compiled = [row["mean_compiled_kl"] for row in summaries]
        random = [row["mean_random_kl"] for row in summaries]
        coverage = [row["coverage"] for row in summaries]
        axis.bar(x - width / 2, compiled, width, label="Compiled")
        axis.bar(x + width / 2, random, width, label="Random")
        axis.set_xticks(x, ["Low", "Medium", "High"])
        axis.set_title(label)
        axis.set_xlabel("KL budget")
        axis.set_ylabel("Mean token KL")
        twin = axis.twinx()
        twin.plot(x, coverage, color="black", marker="o", label="Coverage")
        twin.set_ylim(0, 1.05)
        twin.set_ylabel("Budget coverage")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "prompt_compiler.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "prompt_compiler.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_composition()
    plot_serving()
    plot_ablation()
    plot_prompt_compiler()


if __name__ == "__main__":
    main()
