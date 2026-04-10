from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from aji_engine import AjiEngine, AjiEngineConfig, QwenAjiBridge

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def default_output_path(script_name: str) -> Path:
    return ensure_results_dir() / f"{script_name}.json"


def make_bridge(model_name: str) -> QwenAjiBridge:
    return QwenAjiBridge(model_name=model_name)


def make_engine(
    model_name: str,
    *,
    bridge: QwenAjiBridge | None = None,
    genome_scale: float = 0.02,
    max_nodes: int = 20,
    max_response_tokens: int = 128,
) -> AjiEngine:
    config = AjiEngineConfig(
        model_name=model_name,
        genome_scale=genome_scale,
        max_nodes=max_nodes,
        max_response_tokens=max_response_tokens,
        state_path=None,
        metrics_path=None,
    )
    engine = AjiEngine(config=config, bridge=bridge)
    set_genome_scale(engine.bridge, genome_scale)
    engine._sync_self_to_bridge()
    return engine


def set_genome_scale(bridge: QwenAjiBridge, scale: float) -> None:
    for genome in bridge.genomes.values():
        genome.scale = float(scale)


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def similarity_ratio(left: str, right: str) -> float:
    return float(SequenceMatcher(a=left, b=right).ratio())


def short_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
