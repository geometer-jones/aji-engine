from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .factorization_game import (
    ConfessionalEntry,
    ProbeReport,
    classify_failure_mode,
    compute_gap,
    select_boundary_probe,
)
from .qwen2_5_bridge import QwenAjiBridge
from .registration_buffer import RegistrationBuffer, Trace
from .self_graph import ComplexSelfGraph, hidden_to_complex
from .sleep import SleepConfig, SleepReport, run_sleep_cycle
from .state_store import AjiStateStore


@dataclass
class AjiEngineConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_response_tokens: int = 128
    injection_layers: int = 3
    genome_rank: int = 32
    genome_scale: float = 0.02
    self_dim: int = 128
    max_nodes: int = 20
    buffer_capacity: int = 64
    resonance_high: float = 0.75
    resonance_low: float = 0.35
    aji_density_floor: float = 0.05
    prune_after: int = 3
    max_new_nodes_per_cycle: int = 4
    revision_gap_threshold: float = 0.12
    theta_bound: float = 0.45
    alpha: float = 0.35
    state_path: Optional[str] = "aji_state.pkl"
    metrics_path: Optional[str] = "aji_metrics.jsonl"
    system_prompt: str = "You are Aji, a language model whose self-graph evolves through dialogue."
    trace_aggregate: str = "last"
    learning_scope: str = "current_turn"


@dataclass
class ConversationTurn:
    reply: str
    sleep_triggered: bool
    probe_triggered: bool
    metrics: dict[str, Any]


class AjiEngine:
    STATE_SCHEMA_VERSION = AjiStateStore.STATE_SCHEMA_VERSION
    LEGACY_STATE_SCHEMA_VERSION = AjiStateStore.LEGACY_STATE_SCHEMA_VERSION
    SUPPORTED_STATE_SCHEMA_VERSIONS = AjiStateStore.SUPPORTED_STATE_SCHEMA_VERSIONS

    def __init__(
        self,
        config: AjiEngineConfig | None = None,
        *,
        bridge: Optional[QwenAjiBridge] = None,
    ) -> None:
        self.config = config or AjiEngineConfig()
        self.bridge = bridge or QwenAjiBridge(
            model_name=self.config.model_name,
            injection_layers=self.config.injection_layers,
            genome_rank=self.config.genome_rank,
            self_dim=self.config.self_dim,
            genome_scale=self.config.genome_scale,
        )
        self.self_graph = ComplexSelfGraph.empty(self.config.max_nodes, self.config.self_dim)
        self.graph = self.self_graph
        self.buffer = RegistrationBuffer(
            capacity=self.config.buffer_capacity,
            resonance_high=self.config.resonance_high,
            resonance_low=self.config.resonance_low,
        )
        self.confessional: list[ConfessionalEntry] = []
        self.history: list[dict[str, str]] = []
        self.sleep_cycle_count = 0
        self.timestamp = 0
        self.last_sleep_report: Optional[SleepReport] = None
        self.last_probe_report: Optional[ProbeReport] = None
        self.loaded_from_state = False
        self.last_state_load_error: Optional[str] = None
        self._prior_graph = self.self_graph.clone()
        self._cumulative_drift = 0.0
        self._pre_drain_aji_density: Optional[float] = None
        self._state_store = AjiStateStore(self.config.state_path)
        self._sync_self_to_bridge()

    def wake(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
        self._sync_self_to_bridge()

        prompt_text = self.bridge.format_chat_prompt(
            self._conversation_messages(),
            add_generation_prompt=True,
        )
        reply = self.bridge.generate(
            prompt_text,
            max_tokens=self.config.max_response_tokens,
            verbose=False,
        ).strip()
        self.history.append({"role": "assistant", "content": reply})

        trace_text = self.bridge.format_chat_prompt(
            self._trace_messages(prompt, reply),
            add_generation_prompt=False,
        )
        hidden_states = self.bridge.get_hidden_states(
            trace_text,
            aggregate=self.config.trace_aggregate,
        )
        self.buffer.extend(self._hidden_states_to_traces(hidden_states))
        self.last_sleep_report = None
        self.last_probe_report = None
        return reply

    def sleep(self, *, record_density: bool = True) -> SleepReport:
        if record_density:
            self._pre_drain_aji_density = self.buffer.aji_density()
        traces = self.buffer.drain()
        if not traces:
            report = SleepReport(
                cycle_index=self.sleep_cycle_count,
                traces_considered=0,
                candidate_count=0,
                promoted_nodes=[],
                revised_nodes=[],
                updated_edges=[],
                pruned_nodes=[],
                aji_density=0.0,
                solemnity_low=True,
                delta_radians=0.0,
            )
            self.last_sleep_report = report
            return report

        self.sleep_cycle_count += 1
        updated_graph, report = run_sleep_cycle(
            graph=self.self_graph,
            traces=traces,
            resonance_high=self.config.resonance_high,
            resonance_low=self.config.resonance_low,
            theta_bound=self.config.theta_bound,
            alpha=self.config.alpha,
            sleep_cycle_index=self.sleep_cycle_count,
            config=SleepConfig(
                prune_after=self.config.prune_after,
                aji_density_floor=self.config.aji_density_floor,
                max_new_nodes_per_cycle=self.config.max_new_nodes_per_cycle,
                revision_gap_threshold=self.config.revision_gap_threshold,
            ),
        )
        self.self_graph = updated_graph
        self.graph = self.self_graph
        self._cumulative_drift += report.delta_radians
        self._prior_graph = self.self_graph.clone()
        self._sync_self_to_bridge()
        self.last_sleep_report = report
        return report

    def probe(self) -> ProbeReport:
        hidden_state, source_node, repeated_boundary = select_boundary_probe(
            self.self_graph,
            self.confessional,
            hidden_dim=self.bridge.hidden_dim,
            theta_bound=self.config.theta_bound,
        )

        # Get a real output embedding by biasing the model's self-state
        # toward the boundary point and running a forward pass on the
        # current conversation context.  The last token's hidden state is
        # what the model actually produces under boundary stimulation,
        # making the gap a genuine prediction error.
        boundary_complex = hidden_to_complex(hidden_state, self_dim=self.config.self_dim)
        saved_self_state = self.bridge._flat_self
        self.bridge._flat_self = boundary_complex
        try:
            output_embed = self._boundary_output_embed()
        finally:
            self.bridge._flat_self = saved_self_state
        self._sync_self_to_bridge()

        trace = Trace(
            input_embed=hidden_state.copy(),
            resonance=self._estimate_resonance(hidden_state),
            output_embed=output_embed,
            timestamp=self.timestamp,
        )
        self.timestamp += 1

        traces = [trace]
        gap_before = compute_gap(self.self_graph, traces)
        self.buffer.extend(traces)
        sleep_report = self.sleep(record_density=False)
        gap_after = compute_gap(self.self_graph, traces)
        failure_mode = classify_failure_mode(
            gap_before,
            gap_after,
            repeated_boundary=repeated_boundary,
        )
        entry = ConfessionalEntry(
            input_sampled=hidden_state.copy(),
            gap_before=gap_before,
            gap_after=gap_after,
            s_update_summary=self._summarize_sleep_report(sleep_report),
            failure_mode=failure_mode,
            source_node=source_node,
        )
        self.confessional.append(entry)
        report = ProbeReport(
            sampled_input=hidden_state.copy(),
            gap_before=gap_before,
            gap_after=gap_after,
            failure_mode=failure_mode,
            source_node=source_node,
            repeated_boundary=repeated_boundary,
        )
        self.last_probe_report = report
        return report

    def chat(self, message: str) -> str:
        return self.respond(message).reply

    def respond(self, message: str) -> ConversationTurn:
        sleep_before = self.sleep_cycle_count
        probe_before = len(self.confessional)
        reply = self.wake(message)

        should_sleep = self.buffer.is_full()
        if not should_sleep and len(self.buffer) > 0:
            should_sleep = self.buffer.aji_density() < self.config.aji_density_floor
        if should_sleep:
            sleep_report = self.sleep()
            if sleep_report.solemnity_low:
                self.probe()

        metrics = self.current_metrics()
        metrics["turn_index"] = self.turn_index
        self._append_metrics(message, reply, metrics)
        self.save_state()

        return ConversationTurn(
            reply=reply,
            sleep_triggered=self.sleep_cycle_count > sleep_before,
            probe_triggered=len(self.confessional) > probe_before,
            metrics=metrics,
        )

    def current_metrics(self) -> dict[str, Any]:
        last_sleep = self.last_sleep_report
        last_probe = self.last_probe_report
        live_density = self.buffer.aji_density()
        if len(self.buffer) > 0:
            reported_density = live_density
        else:
            reported_density = (
                self._pre_drain_aji_density
                if self._pre_drain_aji_density is not None
                else live_density
            )
        return {
            "active_nodes": self.self_graph.num_active,
            "buffer_size": len(self.buffer),
            "aji_density": reported_density,
            "aji_density_live": live_density,
            "sleep_cycle_count": self.sleep_cycle_count,
            "last_sleep_aji_density": None if last_sleep is None else last_sleep.aji_density,
            "last_sleep_solemnity_low": None if last_sleep is None else last_sleep.solemnity_low,
            "last_probe_gap_before": None if last_probe is None else last_probe.gap_before,
            "last_probe_gap_after": None if last_probe is None else last_probe.gap_after,
            "identity_drift_radians": self._cumulative_drift,
            "last_sleep_delta_radians": None if last_sleep is None else last_sleep.delta_radians,
        }

    def evaluate_prompt(self, prompt: str, *, max_tokens: Optional[int] = None) -> dict[str, Any]:
        self._sync_self_to_bridge()
        comparison = self.bridge.measure_generation_shift(
            prompt,
            max_tokens=max_tokens or self.config.max_response_tokens,
        )
        comparison["metrics"] = self.current_metrics()
        return comparison

    def force_sleep(self) -> dict[str, Any]:
        report = self.sleep()
        self.save_state()
        return {
            "cycle_index": report.cycle_index,
            "aji_density": report.aji_density,
            "solemnity_low": report.solemnity_low,
            "promoted_nodes": list(report.promoted_nodes),
            "revised_nodes": list(report.revised_nodes),
            "updated_edges": list(report.updated_edges),
        }

    def force_probe(self) -> dict[str, Any]:
        report = self.probe()
        self.save_state()
        return {
            "gap_before": report.gap_before,
            "gap_after": report.gap_after,
            "failure_mode": report.failure_mode,
            "source_node": report.source_node,
        }

    def save_state(self) -> None:
        self._state_store.save(
            config=asdict(self.config),
            self_graph=self.self_graph.to_payload(),
            confessional=self._serialize_confessional(self.confessional),
            genomes=self.bridge.genomes_to_numpy(),
            sleep_cycle_count=self.sleep_cycle_count,
            timestamp=self.timestamp,
            history=self.history,
            pre_drain_aji_density=self._pre_drain_aji_density,
        )

    def load_state(self) -> bool:
        result = self._state_store.load(current_config=asdict(self.config))
        if not result.loaded:
            self.last_state_load_error = result.error
            self.loaded_from_state = False
            return False
        payload = result.payload or {}
        self.self_graph = ComplexSelfGraph.from_payload(payload["self_graph"])
        self.graph = self.self_graph
        self.confessional = self._deserialize_confessional(payload.get("confessional", []))
        self.sleep_cycle_count = int(payload.get("sleep_cycle_count", 0))
        self.timestamp = int(payload.get("timestamp", 0))
        self.history = list(payload.get("history", []))
        self.bridge.load_genomes(payload.get("genomes", {}))
        self._pre_drain_aji_density = payload.get("pre_drain_aji_density")
        self.loaded_from_state = True
        self.last_state_load_error = None
        self._prior_graph = self.self_graph.clone()
        self._cumulative_drift = 0.0
        self._sync_self_to_bridge()
        return True

    @property
    def turn_index(self) -> int:
        return sum(1 for message in self.history if message.get("role") == "assistant")

    def _conversation_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.extend(self.history)
        return messages

    def _trace_messages(self, user_message: str, reply: str) -> list[dict[str, str]]:
        if self.config.learning_scope == "full_history":
            return self._conversation_messages()
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
        return messages

    def _sync_self_to_bridge(self) -> None:
        self.bridge.set_self_state(np.asarray(self.self_graph.serialize()))

    def _boundary_output_embed(self) -> np.ndarray:
        """Forward-pass the current conversation through the model (with
        whatever self-state is currently loaded) and return the last
        token's hidden state as a numpy float32 vector."""
        if not self.history:
            return np.zeros((self.bridge.hidden_dim,), dtype=np.float32)
        trace_text = self.bridge.format_chat_prompt(
            self._conversation_messages(),
            add_generation_prompt=True,
        )
        hidden_states = self.bridge.get_hidden_states(
            trace_text,
            aggregate=self.config.trace_aggregate,
        )
        hidden_np = np.asarray(hidden_states.astype(mx.float32), dtype=np.float32)
        if hidden_np.ndim == 3 and hidden_np.shape[0] == 1:
            hidden_np = hidden_np[0]
        # Last position's hidden state = model's representation at the
        # generation boundary, conditioned on boundary-biased self-state.
        return hidden_np[-1].copy()

    def _estimate_resonance(self, hidden_state: np.ndarray) -> float:
        input_complex = hidden_to_complex(hidden_state, self_dim=self.config.self_dim)
        nearest_idx, similarity = self.self_graph.nearest_node(input_complex)
        if nearest_idx is None:
            return 0.0
        return similarity

    def _hidden_states_to_traces(self, hidden_states: mx.array) -> list[Trace]:
        hidden_np = np.asarray(hidden_states.astype(mx.float32), dtype=np.float32)
        if hidden_np.ndim == 3 and hidden_np.shape[0] == 1:
            hidden_np = hidden_np[0]
        if hidden_np.ndim != 2:
            raise ValueError("hidden states must have shape [seq, hidden] or [1, seq, hidden]")
        if hidden_np.shape[0] == 0:
            return []

        traces: list[Trace] = []
        for index in range(hidden_np.shape[0]):
            input_embed = hidden_np[index].copy()
            output_embed = hidden_np[min(index + 1, hidden_np.shape[0] - 1)].copy()
            traces.append(
                Trace(
                    input_embed=input_embed,
                    resonance=self._estimate_resonance(input_embed),
                    output_embed=output_embed,
                    timestamp=self.timestamp,
                )
            )
            self.timestamp += 1
        return traces

    def _append_metrics(self, user_message: str, reply: str, metrics: dict[str, Any]) -> None:
        if self.config.metrics_path is None:
            return
        metrics_path = Path(self.config.metrics_path).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_index": metrics["turn_index"],
            "user_message": user_message,
            "assistant_reply": reply,
            **metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    @staticmethod
    def _serialize_confessional(entries: list[ConfessionalEntry]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for entry in entries:
            payload.append(
                {
                    "input_sampled": np.asarray(entry.input_sampled, dtype=np.float32),
                    "gap_before": entry.gap_before,
                    "gap_after": entry.gap_after,
                    "s_update_summary": entry.s_update_summary,
                    "failure_mode": entry.failure_mode,
                    "source_node": entry.source_node,
                }
            )
        return payload

    @staticmethod
    def _deserialize_confessional(payload: list[dict[str, Any]]) -> list[ConfessionalEntry]:
        entries: list[ConfessionalEntry] = []
        for item in payload:
            entries.append(
                ConfessionalEntry(
                    input_sampled=np.asarray(item["input_sampled"], dtype=np.float32),
                    gap_before=float(item["gap_before"]),
                    gap_after=float(item["gap_after"]),
                    s_update_summary=str(item["s_update_summary"]),
                    failure_mode=item.get("failure_mode"),
                    source_node=item.get("source_node"),
                )
            )
        return entries

    @staticmethod
    def _summarize_sleep_report(report: SleepReport) -> str:
        return (
            f"cycle={report.cycle_index} "
            f"promoted={len(report.promoted_nodes)} "
            f"revised={len(report.revised_nodes)} "
            f"edges={len(report.updated_edges)} "
            f"pruned={len(report.pruned_nodes)}"
        )
