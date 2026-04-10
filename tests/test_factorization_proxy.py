import mlx.core as mx
import numpy as np
import pytest
from typing import Optional

from aji_engine import (
    ComplexSelfGraph,
    ConfessionalEntry,
    Trace,
    classify_failure_mode,
    complex_angle,
    complex_to_hidden,
    compute_gap,
    hidden_to_complex,
    select_boundary_probe,
)


def _hidden_basis(offset: int) -> np.ndarray:
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[offset] = 1.0
    return hidden


def test_compute_gap_is_zero_for_perfect_prediction() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    node_index = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert node_index is not None

    predicted_hidden = np.asarray(complex_to_hidden(graph.nodes[node_index]), dtype=np.float32)
    trace = Trace(
        input_embed=_hidden_basis(0),
        resonance=0.9,
        output_embed=predicted_hidden,
        timestamp=0,
    )

    assert compute_gap(graph, [trace]) == pytest.approx(0.0, abs=1e-7)


def test_compute_gap_is_large_for_mispredicted_trace() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    node_index = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert node_index is not None

    trace = Trace(
        input_embed=_hidden_basis(0),
        resonance=0.9,
        output_embed=_hidden_basis(1),
        timestamp=0,
    )

    assert compute_gap(graph, [trace]) > 0.18


def test_compute_gap_spreads_familiar_and_novel_inputs() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    node_index = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert node_index is not None

    familiar = Trace(
        input_embed=_hidden_basis(0),
        resonance=0.95,
        output_embed=np.asarray(complex_to_hidden(graph.nodes[node_index]), dtype=np.float32),
        timestamp=0,
    )
    novel = Trace(
        input_embed=_hidden_basis(256),
        resonance=0.0,
        output_embed=_hidden_basis(1),
        timestamp=1,
    )

    familiar_gap = compute_gap(graph, [familiar])
    novel_gap = compute_gap(graph, [novel])

    assert familiar_gap < 0.02
    assert novel_gap > 0.2


@pytest.mark.parametrize(
    ("gap_before", "gap_after", "repeated_boundary", "expected"),
    [
        (0.07, 0.07, False, None),
        (0.10, 0.16, False, "collapsed"),
        (0.10, 0.105, True, "stagnation"),
        (0.10, 0.105, False, "no_progress"),
        (0.20, 0.10, False, None),
    ],
)
def test_classify_failure_mode_paths(
    gap_before: float,
    gap_after: float,
    repeated_boundary: bool,
    expected: Optional[str],
) -> None:
    assert (
        classify_failure_mode(
            gap_before,
            gap_after,
            repeated_boundary=repeated_boundary,
        )
        == expected
    )


def test_select_boundary_probe_returns_nonzero_hidden_probe() -> None:
    theta_bound = 0.45
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    first = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    second = graph.add_node(hidden_to_complex(_hidden_basis(1)))

    assert first is not None
    assert second is not None

    probe, source_node, repeated_boundary = select_boundary_probe(
        graph,
        [],
        hidden_dim=1536,
        theta_bound=theta_bound,
    )
    probe_mx = mx.array(probe)
    probe_complex = hidden_to_complex(probe_mx, self_dim=graph.self_dim)

    assert source_node in {first, second}
    assert repeated_boundary is False
    assert probe.shape == (1536,)
    assert float(np.array(mx.linalg.norm(probe_mx))) > 0.0
    assert complex_angle(graph.nodes[int(source_node)], probe_complex) == pytest.approx(theta_bound, abs=0.05)


def test_select_boundary_probe_generates_a_new_boundary_after_failed_probe() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    node_index = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert node_index is not None

    first_probe, _source_node, _repeated = select_boundary_probe(graph, [], hidden_dim=1536, theta_bound=0.45)
    confessional = [
        ConfessionalEntry(
            input_sampled=first_probe.copy(),
            gap_before=0.2,
            gap_after=0.205,
            s_update_summary="no change",
            failure_mode="stagnation",
            source_node=node_index,
        )
    ]

    second_probe, repeated_source, repeated_boundary = select_boundary_probe(
        graph,
        confessional,
        hidden_dim=1536,
        theta_bound=0.45,
    )

    assert repeated_source == node_index
    assert repeated_boundary is False
    assert np.linalg.norm(second_probe) > 0.0
    assert np.dot(first_probe, second_probe) / (np.linalg.norm(first_probe) * np.linalg.norm(second_probe)) < 0.97
