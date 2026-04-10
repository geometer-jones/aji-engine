from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import mlx.core as mx
import numpy as np

EPS = 1e-6

EDGE_NONE = 0
EDGE_SUPPORTS = 1
EDGE_CONTRADICTS = 2
EDGE_SUBSUMES = 3
EDGE_EXTENDS = 4

_EDGE_PHASES = {
    EDGE_NONE: 0.0,
    EDGE_SUPPORTS: 0.0,
    EDGE_CONTRADICTS: math.pi,
    EDGE_SUBSUMES: math.pi / 2.0,
    EDGE_EXTENDS: -math.pi / 2.0,
}


def _complex_unit(phase: float) -> mx.array:
    return mx.array(np.complex64(math.cos(phase) + 1j * math.sin(phase)))


def _copy_array(value: mx.array) -> mx.array:
    return mx.array(np.array(value))


def _ensure_complex(value: mx.array | np.ndarray | Sequence[complex] | Sequence[float]) -> mx.array:
    return mx.array(value, dtype=mx.complex64)


def _ensure_real(value: mx.array | np.ndarray | Sequence[float]) -> mx.array:
    return mx.array(value, dtype=mx.float32)


def normalize_complex(value: mx.array, axis: int = -1, eps: float = EPS) -> mx.array:
    norms = mx.maximum(mx.linalg.norm(value, axis=axis, keepdims=True), eps)
    return value / norms


def complex_inner(left: mx.array, right: mx.array) -> mx.array:
    return mx.sum(mx.conj(left) * right, axis=-1)


def complex_resonance(left: mx.array, right: mx.array, eps: float = EPS) -> mx.array:
    numerator = mx.abs(complex_inner(left, right))
    left_norm = mx.maximum(mx.linalg.norm(left, axis=-1), eps)
    right_norm = mx.maximum(mx.linalg.norm(right, axis=-1), eps)
    return numerator / (left_norm * right_norm)


def complex_distance(left: mx.array, right: mx.array, eps: float = EPS) -> float:
    resonance = complex_resonance(left.reshape(1, -1), right.reshape(1, -1), eps=eps)
    return float(1.0 - np.array(resonance).reshape(-1)[0])


def complex_angle(left: mx.array, right: mx.array, eps: float = EPS) -> float:
    left_u = normalize_complex(left.reshape(1, -1), eps=eps).reshape(-1)
    right_u = normalize_complex(right.reshape(1, -1), eps=eps).reshape(-1)
    cosine = complex_resonance(left_u.reshape(1, -1), right_u.reshape(1, -1), eps=eps)
    return float(np.arccos(np.clip(np.array(cosine).reshape(-1)[0], -1.0, 1.0)))


def _head_phases(head_count: int) -> mx.array:
    phases = np.exp(2j * np.pi * np.arange(head_count) / max(head_count, 1)).astype(np.complex64)
    return mx.array(phases)


def hidden_to_complex(hidden_state: mx.array | np.ndarray | Sequence[float], *, self_dim: int = 128) -> mx.array:
    hidden = _ensure_real(hidden_state)
    if hidden.shape[-1] % self_dim != 0:
        raise ValueError(f"hidden dimension {hidden.shape[-1]} is not divisible by self_dim={self_dim}")

    head_count = hidden.shape[-1] // self_dim
    heads = hidden.reshape(*hidden.shape[:-1], head_count, self_dim)
    phases = _head_phases(head_count).reshape((1,) * (heads.ndim - 2) + (head_count, 1))
    combined = mx.sum(heads.astype(mx.complex64) * phases, axis=-2)
    return combined / math.sqrt(head_count)


def complex_to_hidden(state: mx.array | np.ndarray | Sequence[complex], *, hidden_dim: int = 1536) -> mx.array:
    flat = _ensure_complex(state)
    self_dim = flat.shape[-1]
    if hidden_dim % self_dim != 0:
        raise ValueError(f"hidden_dim={hidden_dim} is not divisible by self_dim={self_dim}")

    head_count = hidden_dim // self_dim
    phases = mx.conj(_head_phases(head_count)).reshape((1,) * (flat.ndim - 1) + (head_count, 1))
    expanded = flat.reshape(*flat.shape[:-1], 1, self_dim) * phases
    return (mx.real(expanded) / math.sqrt(head_count)).reshape(*flat.shape[:-1], hidden_dim)


def project_within_cone(reference: mx.array, candidate: mx.array, theta_bound: float) -> mx.array:
    reference_u = normalize_complex(reference.reshape(1, -1)).reshape(-1)
    candidate_u = normalize_complex(candidate.reshape(1, -1)).reshape(-1)
    alignment = complex_inner(reference_u.reshape(1, -1), candidate_u.reshape(1, -1)).reshape(())
    alignment_np = np.array(alignment)
    if np.abs(alignment_np) < EPS:
        cosine = 0.0
        aligned = candidate_u
    else:
        phase = alignment / mx.maximum(mx.abs(alignment), EPS)
        aligned = candidate_u * mx.conj(phase)
        cosine = float(np.clip(np.real(np.array(complex_inner(reference_u.reshape(1, -1), aligned.reshape(1, -1))).reshape(-1)[0]), -1.0, 1.0))

    angle = math.acos(cosine)
    if angle <= theta_bound:
        return candidate_u

    orthogonal = aligned - reference_u * cosine
    orth_norm = float(np.array(mx.linalg.norm(orthogonal)))
    if orth_norm < EPS:
        return reference_u
    orthogonal = orthogonal / orth_norm
    projected = reference_u * math.cos(theta_bound) + orthogonal * math.sin(theta_bound)
    return normalize_complex(projected.reshape(1, -1)).reshape(-1)


@dataclass
class ComplexSelfGraph:
    nodes: mx.array
    edges: mx.array
    node_active: mx.array
    node_age: mx.array
    num_active: int

    @classmethod
    def empty(cls, max_nodes: int, self_dim: int) -> "ComplexSelfGraph":
        return cls(
            nodes=mx.zeros((max_nodes, self_dim), dtype=mx.complex64),
            edges=mx.zeros((max_nodes, max_nodes), dtype=mx.int32),
            node_active=mx.zeros((max_nodes,), dtype=mx.bool_),
            node_age=mx.zeros((max_nodes,), dtype=mx.int32),
            num_active=0,
        )

    @property
    def max_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def self_dim(self) -> int:
        return int(self.nodes.shape[1])

    def clone(self) -> "ComplexSelfGraph":
        return ComplexSelfGraph(
            nodes=_copy_array(self.nodes),
            edges=_copy_array(self.edges),
            node_active=_copy_array(self.node_active),
            node_age=_copy_array(self.node_age),
            num_active=self.num_active,
        )

    def active_indices(self) -> list[int]:
        return np.nonzero(np.array(self.node_active))[0].astype(int).tolist()

    def active_nodes(self) -> mx.array:
        indices = self.active_indices()
        if not indices:
            return mx.zeros((0, self.self_dim), dtype=self.nodes.dtype)
        return self.nodes[mx.array(indices, dtype=mx.int32)]

    def resonance(self, query: mx.array, node_idx: int) -> float:
        if not bool(np.array(self.node_active[node_idx])):
            return 0.0
        value = complex_resonance(
            _ensure_complex(query).reshape(1, -1),
            self.nodes[node_idx].reshape(1, -1),
        )
        return float(np.array(value).reshape(-1)[0])

    def nearest_nodes(self, query: mx.array, k: int = 1) -> list[tuple[int, float]]:
        indices = self.active_indices()
        if not indices:
            return []
        active = self.nodes[mx.array(indices, dtype=mx.int32)]
        scores = np.array(complex_resonance(active, _ensure_complex(query).reshape(-1))).reshape(-1)
        order = np.argsort(-scores)[: min(k, len(indices))]
        return [(indices[int(pos)], float(scores[int(pos)])) for pos in order]

    def nearest_node(self, query: mx.array) -> tuple[Optional[int], float]:
        nearest = self.nearest_nodes(query, k=1)
        if not nearest:
            return None, 0.0
        return nearest[0]

    def serialize(self) -> mx.array:
        indices = self.active_indices()
        if not indices:
            return mx.zeros((self.self_dim,), dtype=self.nodes.dtype)

        active = normalize_complex(self.nodes[mx.array(indices, dtype=mx.int32)])
        query = normalize_complex(mx.mean(active, axis=0, keepdims=True))
        node_scores = complex_resonance(active, query.reshape(-1))
        node_weights = mx.softmax(node_scores, axis=0)
        node_summary = mx.sum(active * node_weights.reshape(-1, 1), axis=0)

        edge_reprs: list[mx.array] = []
        edge_matrix = np.array(self.edges)
        for source in indices:
            for target in indices:
                edge_type = int(edge_matrix[source, target])
                if edge_type == EDGE_NONE or source == target:
                    continue
                relation = _complex_unit(_EDGE_PHASES.get(edge_type, 0.0))
                edge_repr = normalize_complex((self.nodes[source] * self.nodes[target] * relation).reshape(1, -1)).reshape(-1)
                edge_reprs.append(edge_repr)

        if edge_reprs:
            edge_stack = mx.stack(edge_reprs, axis=0)
            edge_scores = complex_resonance(edge_stack, query.reshape(-1))
            edge_weights = mx.softmax(edge_scores, axis=0)
            edge_summary = mx.sum(edge_stack * edge_weights.reshape(-1, 1), axis=0)
        else:
            edge_summary = mx.zeros((self.self_dim,), dtype=self.nodes.dtype)

        return normalize_complex((node_summary + edge_summary).reshape(1, -1)).reshape(-1)

    def bind(self, idx_a: int, idx_b: int) -> mx.array:
        if not bool(np.array(self.node_active[idx_a])) or not bool(np.array(self.node_active[idx_b])):
            raise ValueError("bind requires two active nodes")
        return self.nodes[idx_a] * self.nodes[idx_b]

    def superpose(self, indices: Sequence[int]) -> mx.array:
        if not indices:
            return mx.zeros((self.self_dim,), dtype=self.nodes.dtype)
        return mx.sum(self.nodes[mx.array(list(indices), dtype=mx.int32)], axis=0)

    def add_node(self, embedding: mx.array) -> Optional[int]:
        if self.num_active >= self.max_nodes:
            return None
        free = np.nonzero(~np.array(self.node_active))[0]
        if free.size == 0:
            return None
        index = int(free[0])
        self.nodes[index] = normalize_complex(_ensure_complex(embedding).reshape(1, -1)).reshape(-1)
        self.edges[index, :] = 0
        self.edges[:, index] = 0
        self.node_active[index] = True
        self.node_age[index] = 0
        self.num_active += 1
        return index

    def remove_node(self, index: int) -> None:
        if not bool(np.array(self.node_active[index])):
            return
        self.nodes[index] = mx.zeros((self.self_dim,), dtype=self.nodes.dtype)
        self.edges[index, :] = 0
        self.edges[:, index] = 0
        self.node_active[index] = False
        self.node_age[index] = 0
        self.num_active -= 1

    def set_edge(self, source: int, target: int, edge_type: int) -> None:
        if source == target:
            return
        if not bool(np.array(self.node_active[source])) or not bool(np.array(self.node_active[target])):
            return
        self.edges[source, target] = int(edge_type)

    def increment_ages(self) -> None:
        for index in self.active_indices():
            self.node_age[index] = int(np.array(self.node_age[index])) + 1

    def reset_age(self, index: int) -> None:
        if bool(np.array(self.node_active[index])):
            self.node_age[index] = 0

    def to_payload(self) -> dict[str, np.ndarray | int]:
        return {
            "nodes": np.array(self.nodes),
            "edges": np.array(self.edges),
            "node_active": np.array(self.node_active),
            "node_age": np.array(self.node_age),
            "num_active": self.num_active,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, np.ndarray | int]) -> "ComplexSelfGraph":
        return cls(
            nodes=mx.array(payload["nodes"], dtype=mx.complex64),
            edges=mx.array(payload["edges"], dtype=mx.int32),
            node_active=mx.array(payload["node_active"], dtype=mx.bool_),
            node_age=mx.array(payload["node_age"], dtype=mx.int32),
            num_active=int(payload["num_active"]),
        )


SelfGraph = ComplexSelfGraph

