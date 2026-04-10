from __future__ import annotations

from .aji_engine import AjiEngine, AjiEngineConfig, ConversationTurn

QwenAjiRuntimeConfig = AjiEngineConfig


class QwenAjiRuntime(AjiEngine):
    def __init__(self, config: QwenAjiRuntimeConfig | None = None) -> None:
        super().__init__(config=config)
