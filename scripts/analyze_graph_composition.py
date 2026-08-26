import argparse
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr


def composition_metrics(rows, field):
    n_layers = len(rows[0]["bits"])
    singles = {}
    baseline = next(row[field] for row in rows if sum(row["bits"]) == 0)
    for row in rows:
        if sum(row["bits"]) == 1:
            singles[row["bits"].index(1)] = row[field] - baseline
    selected = [row for row in rows if sum(row["bits"]) >= 2]
    actual = np.asarray([row[field] - baseline for row in selected])
    predicted = np.asarray([
        sum(singles.get(layer, 0.0) for layer, bit in enumerate(row["bits"]) if bit)
        for row in selected
    ])
    slope = float(np.dot(predicted, actual) / max(np.dot(predicted, predicted), 1e-30))
    residual = actual - slope * predicted
    return {
        "graphs": len(selected),
        "single_layers_found": len(singles),
        "spearman": float(spearmanr(predicted, actual).statistic),
        "pearson": float(pearsonr(predicted, actual).statistic),
        "slope": slope,
        "relative_rmse": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(actual**2)), 1e-30)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_results")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.graph_results) as handle:
        data = json.load(handle)
    result = {
        "symmetric_kl": composition_metrics(data["rows"], "symmetric_kl"),
        "bpb_degradation": composition_metrics(data["rows"], "bpb_degradation"),
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
