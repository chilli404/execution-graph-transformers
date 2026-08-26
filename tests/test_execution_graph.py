import numpy as np
import pytest

from fogen.execution_graph import (
    additive_effect,
    compile_graph,
    fit_composition_scale,
    single_layer_effects,
)


def test_fit_composition_scale_recovers_subadditive_coefficient():
    assert fit_composition_scale([1, 2, 3], [0.7, 1.4, 2.1]) == pytest.approx(0.7)


def test_single_layer_effects_subtracts_baseline():
    rows = [
        {"bits": [0, 0], "metric": 2.0},
        {"bits": [1, 0], "metric": 2.3},
        {"bits": [0, 1], "metric": 2.5},
    ]

    assert np.allclose(single_layer_effects(rows, "metric"), [0.3, 0.5])


def test_additive_effect_applies_scale():
    assert additive_effect([1, 0, 1], [0.2, 0.3, 0.4], 0.5) == pytest.approx(0.3)


def test_compile_graph_maximizes_saving_under_budget():
    result = compile_graph(
        effect_costs=[0.2, 0.4, 0.7],
        latency_savings=[1.0, 3.0, 5.0],
        budget=0.6,
    )

    assert result["bits"] == [1, 1, 0]
    assert result["predicted_effect"] == pytest.approx(0.6)
    assert result["predicted_saving"] == pytest.approx(4.0)
