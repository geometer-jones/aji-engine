from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

def _flatten_self_state(flat_self: mx.array) -> mx.array:
    return mx.concatenate(
        [
            mx.real(flat_self).astype(mx.float32),
            mx.imag(flat_self).astype(mx.float32),
        ],
        axis=0,
    )


@dataclass
class KVBiasGenome:
    hidden_dim: int = 1536
    kv_dim: int = 256
    # Legacy compatibility field retained so older configs/checkpoints
    # still deserialize, but the runtime path is direct bias rather than
    # low-rank learned projection.
    rank: int = 32
    self_dim: int = 128
    scale: float = 0.02
    key_gain: float = 0.1
    value_gain: float = 1.0

    def _project_self_to_kv(self, flat_self: mx.array) -> mx.array:
        real_self = _flatten_self_state(flat_self).astype(mx.float32)
        width = int(real_self.shape[0])
        if width == self.kv_dim:
            return real_self
        repeats = max(1, (self.kv_dim + width - 1) // width)
        tiled = mx.concatenate([real_self] * repeats, axis=0)
        return tiled[: self.kv_dim]

    def _compute_bias(
        self,
        x: mx.array,
        flat_self: mx.array,
        *,
        target: str,
    ) -> mx.array:
        if self.scale == 0.0:
            return mx.zeros((x.shape[0], x.shape[1], self.kv_dim), dtype=x.dtype)

        if target == "value":
            gain = self.value_gain
        else:
            gain = self.key_gain

        bias = self._project_self_to_kv(flat_self).reshape(1, 1, -1)
        return (bias * (self.scale * gain)).astype(x.dtype)

    def compute_key_bias(self, x: mx.array, flat_self: mx.array) -> mx.array:
        return self._compute_bias(x, flat_self, target="key")

    def compute_value_bias(self, x: mx.array, flat_self: mx.array) -> mx.array:
        return self._compute_bias(x, flat_self, target="value")

    def to_numpy(self) -> dict[str, np.ndarray | float | int]:
        return {
            "hidden_dim": self.hidden_dim,
            "kv_dim": self.kv_dim,
            "rank": self.rank,
            "self_dim": self.self_dim,
            "scale": self.scale,
            "key_gain": self.key_gain,
            "value_gain": self.value_gain,
        }

    @classmethod
    def from_numpy(cls, payload: dict[str, np.ndarray | float | int]) -> "KVBiasGenome":
        return cls(
            hidden_dim=int(payload["hidden_dim"]),
            kv_dim=int(payload["kv_dim"]),
            rank=int(payload.get("rank", 32)),
            self_dim=int(payload["self_dim"]),
            scale=float(payload["scale"]),
            key_gain=float(payload.get("key_gain", 0.1)),
            value_gain=float(payload.get("value_gain", 1.0)),
        )


class KVBiasAttention(nn.Module):
    """Wrap a Qwen attention block and inject a KV bias after projection."""

    def __init__(
        self,
        attention: nn.Module,
        genome: KVBiasGenome,
        flat_self_fn: Callable[[], Optional[mx.array]],
        attention_fn: Optional[Callable[..., mx.array]] = None,
    ) -> None:
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.rope = attention.rope
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads
        self.scale = attention.scale
        self.genome = genome
        self._flat_self_fn = flat_self_fn
        if attention_fn is None:
            from mlx_lm.models.qwen2 import scaled_dot_product_attention

            attention_fn = scaled_dot_product_attention
        self._attention_fn = attention_fn

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _hidden_dim = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        flat_self = self._flat_self_fn()
        if flat_self is not None:
            keys = keys + self.genome.compute_key_bias(x, flat_self).astype(keys.dtype)
            values = values + self.genome.compute_value_bias(x, flat_self).astype(values.dtype)

        queries = queries.reshape(batch_size, sequence_length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = self._attention_fn(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1))


class QwenAjiBridge:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        injection_layers: int = 3,
        genome_rank: int = 32,
        self_dim: int = 128,
        genome_scale: float = 0.02,
    ) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(model_name)
        self.core_model = self.model.model if hasattr(self.model, "model") else self.model
        self.hidden_dim = int(self.model.args.hidden_size)
        self.num_layers = int(self.model.args.num_hidden_layers)
        self.head_dim = int(self.hidden_dim // self.model.args.num_attention_heads)
        self.kv_dim = int(self.model.args.num_key_value_heads * self.head_dim)
        self.self_dim = self_dim
        self.injection_layers = injection_layers
        self.layer_indices = list(range(self.num_layers - injection_layers, self.num_layers))
        self._flat_self: Optional[mx.array] = None

        self.genomes: dict[int, KVBiasGenome] = {
            layer_index: KVBiasGenome(
                hidden_dim=self.hidden_dim,
                kv_dim=self.kv_dim,
                rank=genome_rank,
                self_dim=self.self_dim,
                scale=genome_scale,
            )
            for layer_index in self.layer_indices
        }
        self._patch_attention()

    def set_self_state(self, flat_self: Optional[np.ndarray | Sequence[complex]]) -> None:
        if flat_self is None:
            self._flat_self = None
            return
        self._flat_self = mx.array(np.asarray(flat_self, dtype=np.complex64).reshape(-1))

    def _get_flat_self(self) -> Optional[mx.array]:
        return self._flat_self

    def _patch_attention(self) -> None:
        for layer_index in self.layer_indices:
            attention = self.core_model.layers[layer_index].self_attn
            if isinstance(attention, KVBiasAttention):
                attention.genome = self.genomes[layer_index]
                continue
            self.core_model.layers[layer_index].self_attn = KVBiasAttention(
                attention,
                self.genomes[layer_index],
                self._get_flat_self,
            )

    def format_chat_prompt(
        self,
        messages: Sequence[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        lines: list[str] = []
        for message in messages:
            role = message.get("role", "user").strip().capitalize() or "User"
            lines.append(f"{role}: {message.get('content', '')}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    def generate(self, prompt: str, *, max_tokens: int = 128, verbose: bool = False) -> str:
        from mlx_lm import generate as mlx_generate

        return mlx_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            verbose=verbose,
        )

    def get_hidden_states(
        self,
        text: str,
        *,
        layers: Optional[Sequence[int]] = None,
        aggregate: str = "last",
    ) -> mx.array | dict[int, mx.array]:
        from mlx_lm.models.base import create_attention_mask

        tokens = self.tokenizer.encode(text)
        hidden = self.core_model.embed_tokens(mx.array([tokens]))
        mask = create_attention_mask(hidden, None)
        selected = list(layers) if layers is not None else self.layer_indices
        captured: dict[int, mx.array] = {}

        for index, layer in enumerate(self.core_model.layers):
            hidden = layer(hidden, mask, None)
            if index in selected:
                captured[index] = hidden

        if aggregate == "dict":
            return captured
        ordered = [captured[index] for index in selected if index in captured]
        if not ordered:
            raise ValueError("no hidden states captured for requested layers")
        if aggregate == "last":
            return ordered[-1]
        if aggregate == "mean":
            return mx.mean(mx.stack(ordered, axis=0), axis=0)
        if aggregate == "concat":
            return mx.concatenate(ordered, axis=1)
        raise ValueError(f"unsupported aggregate mode: {aggregate}")

    def genomes_to_numpy(self) -> dict[int, dict[str, np.ndarray | float | int]]:
        return {layer_index: genome.to_numpy() for layer_index, genome in self.genomes.items()}

    def load_genomes(self, payload: dict[int, dict[str, np.ndarray | float | int]]) -> None:
        for key, genome_payload in payload.items():
            layer_index = int(key)
            if layer_index not in self.genomes:
                continue
            self.genomes[layer_index] = KVBiasGenome.from_numpy(genome_payload)
        self._patch_attention()

    def measure_generation_shift(self, prompt: str, *, max_tokens: int = 32) -> dict[str, object]:
        biased = self.generate(prompt, max_tokens=max_tokens, verbose=False)
        saved = self._flat_self
        self._flat_self = None
        try:
            unbiased = self.generate(prompt, max_tokens=max_tokens, verbose=False)
        finally:
            self._flat_self = saved
        return {
            "biased": biased,
            "unbiased": unbiased,
            "different": biased != unbiased,
        }

    def measure_kl_divergence(self, prompt: str, max_tokens: int = 20) -> dict[str, object]:
        return self.measure_generation_shift(prompt, max_tokens=max_tokens)
