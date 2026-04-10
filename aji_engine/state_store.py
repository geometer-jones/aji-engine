from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class StateLoadResult:
    loaded: bool
    error: Optional[str]
    payload: Optional[dict[str, Any]]


class AjiStateStore:
    STATE_SCHEMA_VERSION = 2
    LEGACY_STATE_SCHEMA_VERSION = 1
    SUPPORTED_STATE_SCHEMA_VERSIONS = frozenset({LEGACY_STATE_SCHEMA_VERSION, STATE_SCHEMA_VERSION})

    def __init__(self, state_path: Optional[str]) -> None:
        self._path = None if state_path is None else Path(state_path).expanduser()

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def save(
        self,
        *,
        config: dict[str, Any],
        self_graph: dict[str, Any],
        confessional: list[dict[str, Any]],
        genomes: dict[str, Any],
        sleep_cycle_count: int,
        timestamp: int,
        history: list[dict[str, str]],
        pre_drain_aji_density: Optional[float],
    ) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_schema_version": self.STATE_SCHEMA_VERSION,
            "config": config,
            "self_graph": self_graph,
            "confessional": confessional,
            "genomes": genomes,
            "sleep_cycle_count": sleep_cycle_count,
            "timestamp": timestamp,
            "history": history,
            "pre_drain_aji_density": pre_drain_aji_density,
        }
        with self._path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, *, current_config: dict[str, Any]) -> StateLoadResult:
        if self._path is None or not self._path.exists():
            return StateLoadResult(loaded=False, error=None, payload=None)

        with self._path.open("rb") as handle:
            payload = pickle.load(handle)

        schema_reason = self._state_schema_mismatch(payload)
        if schema_reason is not None:
            return StateLoadResult(loaded=False, error=schema_reason, payload=None)

        mismatch_reason = self._state_config_mismatch(payload.get("config"), current_config)
        if mismatch_reason is not None:
            return StateLoadResult(loaded=False, error=mismatch_reason, payload=None)

        return StateLoadResult(loaded=True, error=None, payload=payload)

    def _state_config_mismatch(
        self,
        saved_config: Any,
        current_config: dict[str, Any],
    ) -> Optional[str]:
        if not isinstance(saved_config, dict):
            return None
        compatibility_fields = {
            "model_name": current_config["model_name"],
            "self_dim": current_config["self_dim"],
            "injection_layers": current_config["injection_layers"],
            "genome_rank": current_config["genome_rank"],
            "max_nodes": current_config["max_nodes"],
        }
        mismatches: list[str] = []
        for field_name, current_value in compatibility_fields.items():
            saved_value = saved_config.get(field_name)
            if saved_value is None or saved_value == current_value:
                continue
            mismatches.append(f"{field_name}: saved={saved_value!r} current={current_value!r}")
        if not mismatches:
            return None
        return "state config mismatch: " + ", ".join(mismatches)

    def _state_schema_mismatch(self, payload: Any) -> Optional[str]:
        schema_version = self._state_schema_version(payload)
        if schema_version in self.SUPPORTED_STATE_SCHEMA_VERSIONS:
            return None
        supported = ", ".join(str(version) for version in sorted(self.SUPPORTED_STATE_SCHEMA_VERSIONS))
        return f"state schema mismatch: saved={schema_version!r} supported=[{supported}]"

    def _state_schema_version(self, payload: Any) -> Optional[int]:
        if not isinstance(payload, dict):
            return None
        saved_version = payload.get("state_schema_version")
        if saved_version is None:
            return self.LEGACY_STATE_SCHEMA_VERSION
        try:
            return int(saved_version)
        except (TypeError, ValueError):
            return None
