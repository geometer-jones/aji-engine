import mlx.core as mx
import numpy as np
import pytest

from aji_engine import (
    EDGE_CONTRADICTS,
    EDGE_EXTENDS,
    EDGE_SUPPORTS,
    ComplexSelfGraph,
    complex_distance,
    complex_resonance,
    hidden_to_complex,
)


def _hidden_basis(offset: int) -> np.ndarray:
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[offset] = 1.0
    return hidden


def _build_graph(edge_types: tuple[int, int, int]) -> ComplexSelfGraph:
    graph = ComplexSelfGraph.empty(max_nodes=6, self_dim=128)
    node_ids = [
        graph.add_node(hidden_to_complex(_hidden_basis(0))),
        graph.add_node(hidden_to_complex(_hidden_basis(1))),
        graph.add_node(hidden_to_complex(_hidden_basis(129))),
    ]
    assert all(node_id is not None for node_id in node_ids)

    first, second, third = [int(node_id) for node_id in node_ids]
    graph.set_edge(first, second, edge_types[0])
    graph.set_edge(second, third, edge_types[1])
    graph.set_edge(third, first, edge_types[2])
    return graph


def test_serialize_resonates_with_each_active_node() -> None:
    graph = _build_graph((EDGE_SUPPORTS, EDGE_CONTRADICTS, EDGE_EXTENDS))

    serialized = graph.serialize()
    mx.eval(serialized)

    assert np.asarray(serialized).shape == (128,)
    assert float(np.array(mx.linalg.norm(serialized))) > 0.0

    for node_index in graph.active_indices():
        resonance = float(
            np.array(
                complex_resonance(
                    graph.nodes[node_index].reshape(1, -1),
                    serialized.reshape(1, -1),
                )
            ).reshape(-1)[0]
        )
        assert resonance > 0.3


def test_serialize_changes_when_only_edge_types_change() -> None:
    graph_a = _build_graph((EDGE_SUPPORTS, EDGE_CONTRADICTS, EDGE_EXTENDS))
    graph_b = _build_graph((EDGE_EXTENDS, EDGE_SUPPORTS, EDGE_CONTRADICTS))

    distance = complex_distance(graph_a.serialize(), graph_b.serialize())

    assert distance > 0.1


def test_payload_round_trip_preserves_serialization_exactly() -> None:
    graph = _build_graph((EDGE_SUPPORTS, EDGE_CONTRADICTS, EDGE_EXTENDS))

    original = graph.serialize()
    restored = ComplexSelfGraph.from_payload(graph.to_payload()).serialize()

    assert complex_distance(original, restored) == pytest.approx(0.0, abs=1e-7)
