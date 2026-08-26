import itertools

import numpy as np


def fit_composition_scale(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    denominator = float(np.dot(predicted, predicted))
    return float(np.dot(predicted, actual) / denominator) if denominator > 0 else 1.0


def single_layer_effects(rows, field):
    n_layers = len(rows[0]["bits"])
    baseline = next(row[field] for row in rows if sum(row["bits"]) == 0)
    effects = np.zeros(n_layers, dtype=float)
    found = np.zeros(n_layers, dtype=bool)
    for row in rows:
        if sum(row["bits"]) == 1:
            layer = row["bits"].index(1)
            effects[layer] = max(0.0, row[field] - baseline)
            found[layer] = True
    if not found.all():
        missing = np.where(~found)[0].tolist()
        raise ValueError(f"Missing single-layer graph effects: {missing}")
    return effects


def additive_effect(bits, effects, scale=1.0):
    return float(scale * np.dot(np.asarray(bits, dtype=float), effects))


def compile_graph(effect_costs, latency_savings, budget, scale=1.0):
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    best = None
    for bits in itertools.product((0, 1), repeat=len(effect_costs)):
        predicted_effect = additive_effect(bits, effect_costs, scale)
        if predicted_effect > budget + 1e-12:
            continue
        predicted_saving = float(np.dot(bits, latency_savings))
        candidate = {
            "bits": list(bits),
            "predicted_effect": predicted_effect,
            "predicted_saving": predicted_saving,
        }
        if best is None or (
            candidate["predicted_saving"], -candidate["predicted_effect"]
        ) > (
            best["predicted_saving"], -best["predicted_effect"]
        ):
            best = candidate
    return best
