import mlx.core as mx
import numpy as np

from aji_engine import AjiEngine, AjiEngineConfig, Trace


class FakeBridge:
    def __init__(self) -> None:
        self.hidden_dim = 1536
        self.turn = 0
        self.latest_self_state = None
        self._flat_self = None

    def set_self_state(self, flat_self) -> None:
        self.latest_self_state = None if flat_self is None else np.asarray(flat_self)

    def format_chat_prompt(self, messages, *, add_generation_prompt=True) -> str:
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nAssistant:"
        return text

    def generate(self, prompt: str, *, max_tokens: int = 128, verbose: bool = False) -> str:
        _ = prompt
        _ = max_tokens
        _ = verbose
        self.turn += 1
        return f"reply-{self.turn}"

    def get_hidden_states(self, text: str, *, layers=None, aggregate: str = "last"):
        _ = text
        _ = layers
        _ = aggregate
        hidden = np.zeros((1, 3, self.hidden_dim), dtype=np.float32)
        base = (self.turn - 1) * 3
        for step in range(3):
            hidden[0, step, (base + step) % self.hidden_dim] = 1.0
        return mx.array(hidden)

    def genomes_to_numpy(self):
        return {}

    def load_genomes(self, payload) -> None:
        _ = payload

    def measure_generation_shift(self, prompt: str, *, max_tokens: int = 32):
        _ = prompt
        _ = max_tokens
        return {"biased": "x", "unbiased": "y", "different": True}


def test_engine_runs_chat_sleep_probe_cycle() -> None:
    engine = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=2,
            max_nodes=6,
            state_path=None,
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    first = engine.respond("first")
    second = engine.respond("second")

    assert first.reply == "reply-1"
    assert second.reply == "reply-2"
    assert engine.sleep_cycle_count >= 2
    assert engine.self_graph.num_active >= 1
    assert np.iscomplexobj(np.asarray(engine.self_graph.serialize()))

    probe = engine.force_probe()
    assert len(engine.confessional) >= 1
    assert probe["gap_before"] >= 0.0
    assert probe["gap_after"] >= 0.0


def test_current_metrics_prefers_live_density_when_buffer_has_pending_traces() -> None:
    engine = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=4,
            max_nodes=4,
            state_path=None,
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    engine._pre_drain_aji_density = 0.0
    engine.buffer.extend(
        [
            Trace(np.array([1.0], dtype=np.float32), 0.4, np.array([1.0], dtype=np.float32), 0),
            Trace(np.array([1.0], dtype=np.float32), 0.6, np.array([1.0], dtype=np.float32), 1),
        ]
    )

    metrics = engine.current_metrics()

    assert metrics["aji_density"] == metrics["aji_density_live"]
    assert metrics["aji_density"] > 0.0


def test_probe_sleep_does_not_overwrite_existing_density_snapshot() -> None:
    engine = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=4,
            max_nodes=4,
            state_path=None,
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    preserved_density = 0.07
    engine._pre_drain_aji_density = preserved_density
    hidden = np.zeros((1536,), dtype=np.float32)
    hidden[0] = 1.0
    engine.buffer.extend([Trace(hidden, 0.0, hidden.copy(), 0)])

    engine.sleep(record_density=False)

    assert engine._pre_drain_aji_density == preserved_density


def test_engine_state_round_trip_loads_when_config_is_compatible(tmp_path) -> None:
    state_path = tmp_path / "aji_state.pkl"
    engine = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=2,
            max_nodes=6,
            state_path=str(state_path),
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    engine.respond("first")
    engine.save_state()

    restored = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=2,
            max_nodes=6,
            state_path=str(state_path),
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    assert restored.load_state() is True
    assert restored.loaded_from_state is True
    assert restored.last_state_load_error is None
    assert restored.turn_index == engine.turn_index
    assert restored.self_graph.num_active == engine.self_graph.num_active


def test_engine_state_load_rejects_incompatible_config(tmp_path) -> None:
    state_path = tmp_path / "aji_state.pkl"
    engine = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=2,
            max_nodes=6,
            self_dim=128,
            state_path=str(state_path),
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )
    engine.respond("first")
    engine.save_state()

    incompatible = AjiEngine(
        AjiEngineConfig(
            buffer_capacity=2,
            max_nodes=6,
            self_dim=64,
            state_path=str(state_path),
            metrics_path=None,
        ),
        bridge=FakeBridge(),
    )

    assert incompatible.load_state() is False
    assert incompatible.loaded_from_state is False
    assert incompatible.last_state_load_error is not None
    assert "self_dim" in incompatible.last_state_load_error
