import itertools

import numpy as np
from scipy.optimize import minimize as _minimize


def fit_composition_scale(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    denominator = float(np.dot(predicted, predicted))
    return float(np.dot(predicted, actual) / denominator) if denominator > 0 else 1.0


def fit_count_adjusted_scale(predicted_sums, actual, n_parallel):
    """Fit the count-adjusted composition model: D(m) ≈ α₀ * Σd * |m|^(-β).

    Returns (alpha0, beta) minimizing MSE over the multi-layer graph set.
    """
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


def compile_graph_greedy(effect_costs, latency_savings, budget, scale=1.0):
    """O(L log L) graph compiler for arbitrary layer counts.

    For uniform latency_savings (all equal), this is provably optimal:
    greedy-by-cost maximizes item count under a weight budget when all
    items have equal value.

    For non-uniform savings, uses greedy-by-ratio (savings/cost),
    which is optimal for the fractional relaxation and a good
    approximation for the 0-1 case. Use compile_graph_dp for exact
    solutions with non-uniform savings.
    """
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    n_layers = len(effect_costs)

    uniform = np.allclose(latency_savings[latency_savings > 0],
                          latency_savings[latency_savings > 0].mean(),
                          rtol=1e-6) if latency_savings.sum() > 0 else True

    if uniform:
        order = np.argsort(effect_costs)
    else:
        ratios = np.where(effect_costs > 1e-12,
                          latency_savings / effect_costs, np.inf)
        order = np.argsort(-ratios)

    bits = [0] * n_layers
    total_cost = 0.0
    for layer in order:
        if latency_savings[layer] <= 0:
            continue
        new_cost = total_cost + effect_costs[layer]
        if scale * new_cost <= budget + 1e-12:
            bits[int(layer)] = 1
            total_cost = new_cost

    return {
        "bits": bits,
        "predicted_effect": float(scale * total_cost),
        "predicted_saving": float(np.dot(bits, latency_savings)),
    }


def compile_graph_dp(effect_costs, latency_savings, budget, scale=1.0,
                     resolution=1000):
    """Exact 0-1 knapsack via DP. O(L * resolution) time and space.

    Handles non-uniform latency savings correctly, unlike greedy-by-ratio.
    Scales to L=128+ in milliseconds (resolution controls budget granularity).
    """
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    n_layers = len(effect_costs)

    scaled_costs = effect_costs * scale
    max_cost = budget
    step = max_cost / resolution
    if step <= 0:
        return {"bits": [0] * n_layers, "predicted_effect": 0.0,
                "predicted_saving": 0.0}

    int_costs = np.round(scaled_costs / step).astype(int)
    int_budget = resolution

    dp = np.full(int_budget + 1, -np.inf)
    dp[0] = 0.0
    choice = np.zeros((n_layers, int_budget + 1), dtype=bool)

    for i in range(n_layers):
        if latency_savings[i] <= 0 or int_costs[i] <= 0:
            continue
        for cap in range(int_budget, int_costs[i] - 1, -1):
            val = dp[cap - int_costs[i]] + latency_savings[i]
            if val > dp[cap]:
                dp[cap] = val
                choice[i, cap] = True

    # Traceback from the capacity with maximum value (not necessarily int_budget)
    bits = [0] * n_layers
    cap = int(np.argmax(dp))
    for i in range(n_layers - 1, -1, -1):
        if choice[i, cap]:
            bits[i] = 1
            cap -= int_costs[i]

    total_cost = sum(effect_costs[i] for i in range(n_layers) if bits[i])
    return {
        "bits": bits,
        "predicted_effect": float(scale * total_cost),
        "predicted_saving": float(np.dot(bits, latency_savings)),
    }
