from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import numpy as np

from .registration_buffer import Trace
from .self_graph import (
    ComplexSelfGraph,
    complex_distance,
    complex_inner,
    complex_to_hidden,
    hidden_to_complex,
    normalize_complex,
)


@dataclass
class ConfessionalEntry:
    input_sampled: np.ndarray
    gap_before: float
    gap_after: float
    s_update_summary: str
    failure_mode: str | None
    source_node: int | None = None


@dataclass
class ProbeReport:
    sampled_input: np.ndarray
    gap_before: float
    gap_after: float
    failure_mode: str | None
    source_node: int | None
    repeated_boundary: bool


def compute_trace_gap(graph: ComplexSelfGraph, trace: Trace) -> float:
    """Calibrated learning gap in the range ~[0.0, 0.3].

    The score blends:
    - input novelty, based on how far the probe is from the nearest node
    - normalized prediction error, comparing nearest-node prediction to a
      structureless centroid baseline

    This keeps familiar, well-modeled traces near 0 while pushing novel or
    badly predicted traces toward 0.3.
    """
    input_complex = hidden_to_complex(trace.input_embed, self_dim=graph.self_dim)
    output_complex = hidden_to_complex(trace.output_embed, self_dim=graph.self_dim)

    centroid = graph.serialize()
    nearest_idx, similarity = graph.nearest_node(input_complex)
    predicted = graph.nodes[nearest_idx] if nearest_idx is not None else centroid

    prediction_error = complex_distance(predicted, output_complex)
    baseline_error = complex_distance(centroid, output_complex)
    if baseline_error > 1e-6:
        normalized_error = min(max(prediction_error / baseline_error, 0.0), 1.0)
    else:
        normalized_error = min(max(prediction_error, 0.0), 1.0)

    novelty = max(0.0, 1.0 - similarity)
    return 0.3 * (0.3 * novelty + 0.7 * normalized_error)


def compute_gap(graph: ComplexSelfGraph, traces: Sequence[Trace]) -> float:
    if not traces:
        return 0.0
    gaps = [compute_trace_gap(graph, trace) for trace in traces]
    return float(sum(gaps) / len(gaps))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _orthogonal_component(node, reference, *, variant: int = 0) -> mx.array:
    width = int(node.shape[0])
    shift = 1 + (variant % max(width - 1, 1))
    phase = np.complex64(np.exp(2j * np.pi * ((variant // max(width - 1, 1)) % 8) / 8.0))
    candidate = mx.roll(mx.conj(node), shift=shift, axis=0) * mx.array(phase)
    if reference is not None and variant % 2 == 1:
        candidate = candidate + mx.roll(reference, shift=min(shift + 1, width - 1), axis=0)
    if float(np.linalg.norm(np.array(candidate))) == 0.0:
        candidate = mx.ones_like(node)
    projection = complex_inner(node.reshape(1, -1), candidate.reshape(1, -1))
    projection = projection / mx.maximum(mx.abs(complex_inner(node.reshape(1, -1), node.reshape(1, -1))), 1e-6)
    orthogonal = candidate - node * projection.reshape(())
    if reference is not None:
        reference_projection = complex_inner(reference.reshape(1, -1), orthogonal.reshape(1, -1))
        orthogonal = orthogonal - reference * reference_projection.reshape(())
    if float(np.linalg.norm(np.array(orthogonal))) == 0.0:
        orthogonal = mx.roll(node, shift=3, axis=0)
    return normalize_complex(orthogonal.reshape(1, -1)).reshape(-1)


def _boundary_hidden(
    node,
    *,
    reference,
    theta_bound: float,
    hidden_dim: int,
    variant: int = 0,
) -> np.ndarray:
    orthogonal = _orthogonal_component(node, reference, variant=variant)
    boundary = normalize_complex(
        (node * math.cos(theta_bound) + orthogonal * math.sin(theta_bound)).reshape(1, -1)
    ).reshape(-1)
    return np.asarray(complex_to_hidden(boundary, hidden_dim=hidden_dim), dtype=np.float32)


def select_boundary_probe(
    graph: ComplexSelfGraph,
    confessional: Sequence[ConfessionalEntry],
    *,
    hidden_dim: int = 1536,
    theta_bound: float = 0.45,
) -> tuple[np.ndarray, int | None, bool]:
    failed_inputs = [
        entry.input_sampled
        for entry in confessional
        if entry.failure_mode in {"collapsed", "no_progress", "stagnation"}
    ]
    failure_counts: dict[int, int] = {}
    for entry in confessional:
        if entry.failure_mode not in {"collapsed", "no_progress", "stagnation"}:
            continue
        if entry.source_node is None:
            continue
        failure_counts[entry.source_node] = failure_counts.get(entry.source_node, 0) + 1
    if graph.num_active == 0:
        return np.zeros((hidden_dim,), dtype=np.float32), None, False

    active = graph.active_indices()
    ordered = sorted(
        active,
        key=lambda idx: (failure_counts.get(idx, 0), -int(np.array(graph.node_age[idx]))),
    )
    reference = graph.serialize()
    fallback: tuple[np.ndarray, int | None, bool] | None = None
    max_variants_per_node = max(4, len(active))

    for node_index in ordered:
        start_variant = failure_counts.get(node_index, 0)
        for offset in range(max_variants_per_node):
            hidden = _boundary_hidden(
                graph.nodes[node_index],
                reference=reference,
                theta_bound=theta_bound,
                hidden_dim=hidden_dim,
                variant=start_variant + offset,
            )
            repeated = any(_cosine_similarity(hidden, past.astype(np.float32)) > 0.97 for past in failed_inputs)
            if fallback is None:
                fallback = (hidden, node_index, True)
            if not repeated:
                return hidden, node_index, False

    if fallback is not None:
        return fallback
    return np.zeros((hidden_dim,), dtype=np.float32), None, True


def classify_failure_mode(
    gap_before: float,
    gap_after: float,
    *,
    repeated_boundary: bool,
) -> str | None:
    if gap_after <= 0.08:
        return None
    if gap_after > gap_before + 0.05:
        return "collapsed"
    if abs(gap_after - gap_before) < 0.01:
        return "stagnation" if repeated_boundary else "no_progress"
    return None
