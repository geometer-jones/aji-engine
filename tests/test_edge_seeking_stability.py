import mlx.core as mx
import numpy as np
import pytest

from aji_engine import (
    EDGE_CONTRADICTS,
    EDGE_SUBSUMES,
    EDGE_SUPPORTS,
    ComplexSelfGraph,
    SleepConfig,
    Trace,
    hidden_to_complex,
    project_within_cone,
    run_sleep_cycle,
)
from aji_engine.self_graph import complex_inner, normalize_complex
from aji_engine.sleep import _infer_edge_type, graph_shift_radians


def _hidden_basis(offset: int) -> np.ndarray:
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[offset] = 1.0
    return hidden


def test_infer_edge_type_is_deterministic() -> None:
    source = hidden_to_complex(_hidden_basis(0))
    target = hidden_to_complex(_hidden_basis(0))

    outputs = [
        _infer_edge_type(
            source,
            target,
            trace_resonance=0.9,
            resonance_high=0.8,
            resonance_low=0.2,
        )
        for _ in range(10)
    ]

    assert outputs == [outputs[0]] * 10


def test_project_within_cone_is_idempotent() -> None:
    theta_bound = 0.45
    reference = hidden_to_complex(_hidden_basis(0))
    candidate = hidden_to_complex(_hidden_basis(1))

    first_projection = project_within_cone(reference, candidate, theta_bound)
    second_projection = project_within_cone(reference, first_projection, theta_bound)

    assert np.allclose(np.asarray(first_projection), np.asarray(second_projection), atol=1e-6)
    assert complex_inner(
        normalize_complex(reference.reshape(1, -1)),
        normalize_complex(second_projection.reshape(1, -1)),
    ).shape == (1,)


def test_consecutive_sleep_cycles_converge() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=6, self_dim=128)
    node_index = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert node_index is not None

    traces = [
        Trace(
            input_embed=_hidden_basis(0),
            resonance=0.4,
            output_embed=_hidden_basis(1),
            timestamp=0,
        ),
        Trace(
            input_embed=_hidden_basis(0),
            resonance=0.45,
            output_embed=_hidden_basis(2),
            timestamp=1,
        ),
    ]

    config = SleepConfig(
        prune_after=3,
        aji_density_floor=0.01,
        max_new_nodes_per_cycle=2,
        revision_gap_threshold=0.18,
    )
    graph_after_first, first_report = run_sleep_cycle(
        graph=graph,
        traces=traces,
        resonance_high=0.8,
        resonance_low=0.2,
        theta_bound=0.45,
        alpha=0.35,
        sleep_cycle_index=1,
        config=config,
    )
    graph_after_second, second_report = run_sleep_cycle(
        graph=graph_after_first,
        traces=traces,
        resonance_high=0.8,
        resonance_low=0.2,
        theta_bound=0.45,
        alpha=0.35,
        sleep_cycle_index=2,
        config=config,
    )

    mx.eval(graph_after_first.serialize(), graph_after_second.serialize())

    assert second_report.delta_radians <= first_report.delta_radians + 1e-6
    assert graph_shift_radians(graph_after_first, graph_after_second) == pytest.approx(
        second_report.delta_radians,
        abs=1e-6,
    )


def test_infer_edge_type_matches_basic_geometry() -> None:
    source = hidden_to_complex(_hidden_basis(0))
    positive_target = hidden_to_complex(_hidden_basis(0))
    negative_target = -positive_target

    positive_inner = np.real(
        np.array(
            complex_inner(
                normalize_complex(source.reshape(1, -1)),
                normalize_complex(positive_target.reshape(1, -1)),
            )
        ).reshape(-1)[0]
    )
    negative_inner = np.real(
        np.array(
            complex_inner(
                normalize_complex(source.reshape(1, -1)),
                normalize_complex(negative_target.reshape(1, -1)),
            )
        ).reshape(-1)[0]
    )

    positive_edge = _infer_edge_type(
        source,
        positive_target,
        trace_resonance=0.9,
        resonance_high=0.8,
        resonance_low=0.2,
    )
    negative_edge = _infer_edge_type(
        source,
        negative_target,
        trace_resonance=0.9,
        resonance_high=0.8,
        resonance_low=0.2,
    )

    assert positive_inner > 0.9
    assert negative_inner < -0.15
    assert positive_edge in {EDGE_SUPPORTS, EDGE_SUBSUMES}
    assert negative_edge == EDGE_CONTRADICTS
