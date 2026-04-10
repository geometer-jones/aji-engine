import math

import numpy as np

from aji_engine import (
    ComplexSelfGraph,
    RegistrationBuffer,
    SleepConfig,
    Trace,
    complex_angle,
    complex_to_hidden,
    hidden_to_complex,
    run_sleep_cycle,
)
from aji_engine.self_graph import normalize_complex


def _hidden_basis(offset: int) -> np.ndarray:
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[offset] = 1.0
    return hidden


def test_sleep_promotes_new_structure_and_prunes_stale_nodes() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    first = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    stale = graph.add_node(hidden_to_complex(_hidden_basis(1)))
    assert first is not None
    assert stale is not None
    graph.node_age[stale] = 3

    traces = [
        Trace(
            input_embed=_hidden_basis(2),
            resonance=0.0,
            output_embed=_hidden_basis(2),
            timestamp=0,
        ),
    ]

    updated_graph, report = run_sleep_cycle(
        graph=graph,
        traces=traces,
        resonance_high=0.8,
        resonance_low=0.2,
        theta_bound=0.45,
        alpha=0.35,
        sleep_cycle_index=1,
        config=SleepConfig(prune_after=2, aji_density_floor=0.01, max_new_nodes_per_cycle=2),
    )

    assert updated_graph.num_active >= 2
    assert report.promoted_nodes
    assert stale in report.pruned_nodes


def test_sleep_new_nodes_anchor_to_nearest_frontier_node() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    anchor = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    other = graph.add_node(hidden_to_complex(_hidden_basis(1)))
    assert anchor is not None
    assert other is not None

    theta_bound = 0.45
    anchor_node = graph.nodes[anchor]
    frontier_direction = normalize_complex(hidden_to_complex(_hidden_basis(2)).reshape(1, -1)).reshape(-1)
    input_complex = normalize_complex(
        (anchor_node * math.cos(0.35) + frontier_direction * math.sin(0.35)).reshape(1, -1)
    ).reshape(-1)
    trace = Trace(
        input_embed=np.asarray(complex_to_hidden(input_complex), dtype=np.float32),
        resonance=0.4,
        output_embed=np.asarray(complex_to_hidden(anchor_node), dtype=np.float32),
        timestamp=0,
    )

    updated_graph, report = run_sleep_cycle(
        graph=graph,
        traces=[trace],
        resonance_high=0.99,
        resonance_low=0.95,
        theta_bound=theta_bound,
        alpha=0.35,
        sleep_cycle_index=1,
        config=SleepConfig(prune_after=3, aji_density_floor=0.01, max_new_nodes_per_cycle=1),
    )

    assert report.promoted_nodes

    new_node = updated_graph.nodes[report.promoted_nodes[0]]
    centroid = graph.serialize()

    assert complex_angle(graph.nodes[anchor], new_node) < theta_bound
    assert complex_angle(centroid, new_node) > theta_bound + 0.1


def test_sleep_learns_from_overflow_summary_replay() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    anchor = graph.add_node(hidden_to_complex(_hidden_basis(0)))
    assert anchor is not None

    buffer = RegistrationBuffer(
        capacity=1,
        resonance_high=0.8,
        resonance_low=0.2,
        histogram_bins=4,
    )
    novel = Trace(
        input_embed=_hidden_basis(2),
        resonance=0.0,
        output_embed=_hidden_basis(2),
        timestamp=0,
    )
    familiar = Trace(
        input_embed=_hidden_basis(0),
        resonance=0.95,
        output_embed=np.asarray(complex_to_hidden(graph.nodes[anchor]), dtype=np.float32),
        timestamp=1,
    )
    buffer.extend([novel, familiar])

    drained = buffer.drain()
    updated_graph, report = run_sleep_cycle(
        graph=graph,
        traces=drained,
        resonance_high=0.8,
        resonance_low=0.2,
        theta_bound=0.45,
        alpha=0.35,
        sleep_cycle_index=1,
        config=SleepConfig(prune_after=3, aji_density_floor=0.01, max_new_nodes_per_cycle=1),
    )

    assert len(drained) == 2
    assert report.promoted_nodes
    assert updated_graph.num_active == 2
