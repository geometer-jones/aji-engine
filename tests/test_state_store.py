import pickle

from aji_engine.state_store import AjiStateStore


def _config(*, self_dim: int = 128) -> dict[str, object]:
    return {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "self_dim": self_dim,
        "injection_layers": 3,
        "genome_rank": 32,
        "max_nodes": 20,
    }


def test_state_store_round_trip_is_versioned(tmp_path) -> None:
    state_path = tmp_path / "aji_state.pkl"
    store = AjiStateStore(str(state_path))
    config = _config()

    store.save(
        config=config,
        self_graph={"nodes": [], "edges": []},
        confessional=[],
        genomes={"scale": 0.02},
        sleep_cycle_count=3,
        timestamp=7,
        history=[{"role": "user", "content": "hi"}],
        pre_drain_aji_density=0.05,
    )

    with state_path.open("rb") as handle:
        raw_payload = pickle.load(handle)

    assert raw_payload["state_schema_version"] == AjiStateStore.STATE_SCHEMA_VERSION

    result = store.load(current_config=config)

    assert result.loaded is True
    assert result.error is None
    assert result.payload is not None
    assert result.payload["timestamp"] == 7


def test_state_store_accepts_legacy_unversioned_payload(tmp_path) -> None:
    state_path = tmp_path / "aji_state.pkl"
    config = _config()
    payload = {
        "config": config,
        "self_graph": {"nodes": [], "edges": []},
        "confessional": [],
        "genomes": {},
        "sleep_cycle_count": 0,
        "timestamp": 1,
        "history": [],
        "pre_drain_aji_density": None,
    }

    with state_path.open("wb") as handle:
        pickle.dump(payload, handle)

    result = AjiStateStore(str(state_path)).load(current_config=config)

    assert result.loaded is True
    assert result.error is None


def test_state_store_rejects_unsupported_schema_version(tmp_path) -> None:
    state_path = tmp_path / "aji_state.pkl"
    config = _config()
    payload = {
        "state_schema_version": AjiStateStore.STATE_SCHEMA_VERSION + 1,
        "config": config,
        "self_graph": {"nodes": [], "edges": []},
        "confessional": [],
        "genomes": {},
        "sleep_cycle_count": 0,
        "timestamp": 1,
        "history": [],
        "pre_drain_aji_density": None,
    }

    with state_path.open("wb") as handle:
        pickle.dump(payload, handle)

    result = AjiStateStore(str(state_path)).load(current_config=config)

    assert result.loaded is False
    assert result.payload is None
    assert result.error is not None
    assert "state schema mismatch" in result.error
