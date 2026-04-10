import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import scaled_dot_product_attention

from aji_engine.qwen2_5_bridge import KVBiasAttention, KVBiasGenome


class IdentityProj(nn.Module):
    def __call__(self, x):
        return x


class IdentityRope(nn.Module):
    def __call__(self, x, offset=0):
        _ = offset
        return x


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = IdentityProj()
        self.k_proj = IdentityProj()
        self.v_proj = IdentityProj()
        self.o_proj = IdentityProj()
        self.rope = IdentityRope()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.scale = 1.0

    def __call__(self, x, mask=None, cache=None):
        batch_size, sequence_length, _hidden_dim = x.shape
        queries = self.q_proj(x).reshape(batch_size, sequence_length, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1))


def _make_genome(scale: float) -> KVBiasGenome:
    return KVBiasGenome(
        hidden_dim=4,
        kv_dim=4,
        rank=2,
        self_dim=2,
        scale=scale,
        key_gain=0.1,
        value_gain=1.0,
    )


def test_attention_wrapper_changes_output_when_scale_is_high() -> None:
    x = mx.array(
        [
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
            ]
        ]
    )
    flat_self = mx.array([1.0 + 1.0j, 1.0 - 1.0j], dtype=mx.complex64)

    baseline = FakeAttention()(x)
    biased = KVBiasAttention(FakeAttention(), _make_genome(scale=5.0), lambda: flat_self)(x)
    mx.eval(baseline, biased)

    assert not np.allclose(np.array(baseline), np.array(biased))


def test_attention_wrapper_respects_direct_bias_gains() -> None:
    x = mx.array(
        [
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
            ]
        ]
    )
    flat_self = mx.array([1.0 + 1.0j, 1.0 - 1.0j], dtype=mx.complex64)

    baseline = FakeAttention()(x)

    zero_genome = _make_genome(scale=5.0)
    zero_genome.key_gain = 0.0
    zero_genome.value_gain = 0.0
    no_bias = KVBiasAttention(FakeAttention(), zero_genome, lambda: flat_self)(x)

    active_bias = KVBiasAttention(FakeAttention(), _make_genome(scale=5.0), lambda: flat_self)(x)
    mx.eval(baseline, no_bias, active_bias)

    np.testing.assert_allclose(np.array(no_bias), np.array(baseline), rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.array(active_bias), np.array(baseline))


def test_genome_load_ignores_legacy_low_rank_fields() -> None:
    payload = {
        "hidden_dim": 4,
        "kv_dim": 4,
        "rank": 2,
        "self_dim": 2,
        "scale": 0.5,
        "A_k": np.ones((4, 2), dtype=np.float32),
        "B_k": np.ones((2, 4), dtype=np.float32),
        "G_k": np.ones((4, 2), dtype=np.float32),
        "A_v": np.ones((4, 2), dtype=np.float32),
        "B_v": np.ones((2, 4), dtype=np.float32),
        "G_v": np.ones((4, 2), dtype=np.float32),
    }

    genome = KVBiasGenome.from_numpy(payload)

    assert genome.hidden_dim == 4
    assert genome.kv_dim == 4
    assert genome.rank == 2
    assert genome.scale == 0.5
    assert genome.key_gain == 0.1
    assert genome.value_gain == 1.0
