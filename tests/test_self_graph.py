import math

import numpy as np

from aji_engine import (
    EDGE_SUPPORTS,
    ComplexSelfGraph,
    complex_angle,
    complex_to_hidden,
    hidden_to_complex,
    project_within_cone,
)


def _hidden_basis(offset: int) -> np.ndarray:
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[offset] = 1.0
    return hidden


def test_complex_self_graph_add_remove_serialize_and_compose() -> None:
    graph = ComplexSelfGraph.empty(max_nodes=4, self_dim=128)
    node_a = hidden_to_complex(_hidden_basis(0))
    node_b = hidden_to_complex(_hidden_basis(128))

    first = graph.add_node(node_a)
    second = graph.add_node(node_b)

    assert first == 0
    assert second == 1
    assert graph.num_active == 2

    graph.set_edge(first, second, EDGE_SUPPORTS)
    serialized = np.asarray(graph.serialize())

    assert serialized.shape == (128,)
    assert np.linalg.norm(serialized) > 0.0
    assert graph.resonance(node_a, first) > 0.99

    nearest_idx, similarity = graph.nearest_node(node_a)
    assert nearest_idx == first
    assert similarity > 0.99

    bound = np.asarray(graph.bind(first, second))
    superposed = np.asarray(graph.superpose([first, second]))
    assert bound.shape == (128,)
    assert superposed.shape == (128,)

    graph.remove_node(first)
    assert graph.num_active == 1
    assert not bool(np.asarray(graph.node_active[first]))
    assert int(np.asarray(graph.edges[:, first]).sum()) == 0
    assert int(np.asarray(graph.edges[first, :]).sum()) == 0


def test_hidden_projection_and_cone_projection_are_shape_stable() -> None:
    state = hidden_to_complex(_hidden_basis(256))
    decoded = np.asarray(complex_to_hidden(state))
    assert decoded.shape == (1536,)

    candidate = state * np.complex64(1.0 + 1.0j)
    projected = project_within_cone(state, candidate, 0.1)
    assert np.asarray(projected).shape == (128,)
    assert complex_angle(state, projected) <= 0.100001 + 1e-6
