from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .registration_buffer import Trace
from .factorization_game import compute_trace_gap
from .self_graph import (
    EDGE_CONTRADICTS,
    EDGE_EXTENDS,
    EDGE_SUBSUMES,
    EDGE_SUPPORTS,
    ComplexSelfGraph,
    complex_angle,
    complex_inner,
    hidden_to_complex,
    normalize_complex,
    project_within_cone,
)


@dataclass
class SleepConfig:
    prune_after: int = 3
    aji_density_floor: float = 0.05
    max_new_nodes_per_cycle: int = 4
    revision_gap_threshold: float = 0.12


@dataclass
class SleepReport:
    cycle_index: int
    traces_considered: int
    candidate_count: int
    promoted_nodes: list[int]
    revised_nodes: list[int]
    updated_edges: list[tuple[int, int, int]]
    pruned_nodes: list[int]
    aji_density: float
    solemnity_low: bool
    delta_radians: float


@dataclass
class _Candidate:
    trace: Trace
    input_complex: object
    output_complex: object
    nearest_idx: int | None
    second_idx: int | None
    similarity: float
    second_similarity: float
    gap: float
    novelty: float
    action: str


def compute_aji_density(
    traces: Sequence[Trace],
    *,
    resonance_high: float,
    resonance_low: float,
) -> float:
    if not traces:
        return 0.0
    midpoint = 0.5 * (resonance_high + resonance_low)
    friction = [
        trace
        for trace in traces
        if resonance_low < trace.resonance < resonance_high
    ]
    if not friction:
        return 0.0
    total = sum(abs(trace.resonance - midpoint) for trace in friction)
    return total / len(traces)


def graph_shift_radians(reference: ComplexSelfGraph, candidate: ComplexSelfGraph) -> float:
    return complex_angle(reference.serialize(), candidate.serialize())


def _infer_edge_type(
    source,
    target,
    *,
    trace_resonance: float,
    resonance_high: float,
    resonance_low: float,
) -> int:
    source_u = normalize_complex(source.reshape(1, -1)).reshape(-1)
    target_u = normalize_complex(target.reshape(1, -1)).reshape(-1)
    similarity = complex_inner(source_u.reshape(1, -1), target_u.reshape(1, -1))
    similarity_np = np.array(similarity).reshape(-1)[0]
    midpoint = 0.5 * (resonance_high + resonance_low)
    if float(np.real(similarity_np)) < -0.15:
        return EDGE_CONTRADICTS
    if trace_resonance < midpoint:
        return EDGE_EXTENDS
    if float(np.abs(similarity_np)) > 0.92:
        return EDGE_SUBSUMES
    return EDGE_SUPPORTS


def _successor_effectiveness(graph: ComplexSelfGraph, node_index: int) -> float:
    scores: list[float] = []
    edges = np.array(graph.edges)
    for target in graph.active_indices():
        if edges[node_index, target] == 0:
            continue
        scores.append(graph.resonance(graph.nodes[node_index], target))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def run_sleep_cycle(
    *,
    graph: ComplexSelfGraph,
    traces: Sequence[Trace],
    resonance_high: float,
    resonance_low: float,
    theta_bound: float,
    alpha: float,
    sleep_cycle_index: int,
    config: SleepConfig | None = None,
) -> tuple[ComplexSelfGraph, SleepReport]:
    sleep_config = config or SleepConfig()
    reference = graph.clone()
    working = graph.clone()
    working.increment_ages()

    candidates: list[_Candidate] = []
    touched_nodes: set[int] = set()

    for trace in traces:
        input_complex = hidden_to_complex(trace.input_embed, self_dim=working.self_dim)
        output_complex = hidden_to_complex(trace.output_embed, self_dim=working.self_dim)
        nearest = working.nearest_nodes(input_complex, k=2)
        nearest_idx = nearest[0][0] if nearest else None
        similarity = nearest[0][1] if nearest else 0.0
        second_idx = nearest[1][0] if len(nearest) > 1 else None
        second_similarity = nearest[1][1] if len(nearest) > 1 else 0.0
        if nearest_idx is not None:
            touched_nodes.add(nearest_idx)

        predicted = working.nodes[nearest_idx] if nearest_idx is not None else working.serialize()
        gap = compute_trace_gap(working, trace)
        novelty = max(0.0, 1.0 - similarity)
        carries_new_information = similarity < resonance_high or gap > sleep_config.revision_gap_threshold
        if not carries_new_information:
            continue

        if nearest_idx is None or similarity < resonance_low:
            action = "add_node"
        elif gap > sleep_config.revision_gap_threshold:
            action = "revise_node"
        elif second_idx is not None and second_similarity > resonance_low:
            action = "add_edge"
        else:
            action = "revise_node"

        candidates.append(
            _Candidate(
                trace=trace,
                input_complex=input_complex,
                output_complex=output_complex,
                nearest_idx=nearest_idx,
                second_idx=second_idx,
                similarity=similarity,
                second_similarity=second_similarity,
                gap=gap,
                novelty=novelty,
                action=action,
            )
        )

    for node_index in touched_nodes:
        working.reset_age(node_index)

    promoted_nodes: list[int] = []
    revised_nodes: list[int] = []
    updated_edges: list[tuple[int, int, int]] = []
    new_nodes_used = 0

    candidates.sort(key=lambda item: (item.novelty + item.gap), reverse=True)
    for candidate in candidates:
        tentative = working.clone()

        if candidate.action == "add_node":
            if new_nodes_used >= sleep_config.max_new_nodes_per_cycle:
                continue
            if candidate.nearest_idx is not None and candidate.second_idx is not None and candidate.second_similarity > resonance_low:
                seed = tentative.bind(candidate.nearest_idx, candidate.second_idx)
                candidate_node = normalize_complex((seed + candidate.output_complex).reshape(1, -1)).reshape(-1)
            else:
                candidate_node = normalize_complex((candidate.input_complex + candidate.output_complex).reshape(1, -1)).reshape(-1)

            reference_flat = reference.serialize()
            anchor = tentative.nodes[candidate.nearest_idx] if candidate.nearest_idx is not None else reference_flat
            if float(np.array(np.linalg.norm(np.array(anchor)))) > 0.0:
                candidate_node = project_within_cone(anchor, candidate_node, theta_bound)

            new_index = tentative.add_node(candidate_node)
            if new_index is None:
                continue
            tentative.reset_age(new_index)
            if candidate.nearest_idx is not None:
                edge_type = _infer_edge_type(
                    tentative.nodes[candidate.nearest_idx],
                    tentative.nodes[new_index],
                    trace_resonance=candidate.trace.resonance,
                    resonance_high=resonance_high,
                    resonance_low=resonance_low,
                )
                tentative.set_edge(candidate.nearest_idx, new_index, edge_type)
            working = tentative
            promoted_nodes.append(new_index)
            new_nodes_used += 1
            if candidate.nearest_idx is not None:
                updated_edges.append(
                    (
                        candidate.nearest_idx,
                        new_index,
                        int(np.array(working.edges[candidate.nearest_idx, new_index])),
                    )
                )
            continue

        if candidate.nearest_idx is None:
            continue

        if candidate.action == "revise_node":
            current = tentative.nodes[candidate.nearest_idx]
            alignment = complex_inner(current.reshape(1, -1), candidate.output_complex.reshape(1, -1))
            alignment_np = np.array(alignment).reshape(-1)[0]
            if abs(alignment_np) < 1e-6:
                aligned = candidate.output_complex
            else:
                phase = alignment / np.float32(max(abs(alignment_np), 1e-6))
                aligned = candidate.output_complex * np.conj(np.array(phase))
            candidate_node = normalize_complex((current + alpha * max(candidate.similarity, 0.1) * aligned).reshape(1, -1)).reshape(-1)
            tentative.nodes[candidate.nearest_idx] = project_within_cone(current, candidate_node, theta_bound)
            tentative.reset_age(candidate.nearest_idx)
            working = tentative
            if candidate.nearest_idx not in revised_nodes:
                revised_nodes.append(candidate.nearest_idx)
            continue

        if candidate.second_idx is None:
            continue

        edge_type = _infer_edge_type(
            tentative.nodes[candidate.nearest_idx],
            tentative.nodes[candidate.second_idx],
            trace_resonance=candidate.trace.resonance,
            resonance_high=resonance_high,
            resonance_low=resonance_low,
        )
        tentative.set_edge(candidate.nearest_idx, candidate.second_idx, edge_type)
        tentative.reset_age(candidate.nearest_idx)
        tentative.reset_age(candidate.second_idx)
        working = tentative
        updated_edges.append((candidate.nearest_idx, candidate.second_idx, edge_type))

    pruned_nodes: list[int] = []
    for node_index in list(working.active_indices()):
        if int(np.array(working.node_age[node_index])) < sleep_config.prune_after:
            continue
        if _successor_effectiveness(working, node_index) >= resonance_low:
            continue
        working.remove_node(node_index)
        pruned_nodes.append(node_index)

    aji_density = compute_aji_density(
        traces,
        resonance_high=resonance_high,
        resonance_low=resonance_low,
    )
    report = SleepReport(
        cycle_index=sleep_cycle_index,
        traces_considered=len(traces),
        candidate_count=len(candidates),
        promoted_nodes=promoted_nodes,
        revised_nodes=revised_nodes,
        updated_edges=updated_edges,
        pruned_nodes=pruned_nodes,
        aji_density=aji_density,
        solemnity_low=aji_density < sleep_config.aji_density_floor,
        delta_radians=graph_shift_radians(reference, working),
    )
    return working, report
