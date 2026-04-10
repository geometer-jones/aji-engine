import numpy as np

import mlx.core as mx
import mlx.nn as nn

from aji_engine.qwen2_5_bridge import KVBiasAttention, KVBiasGenome


class IdentityModule(nn.Module):
    def __call__(self, x, offset=None):
        _ = offset
        return x


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = IdentityModule()
        self.k_proj = IdentityModule()
        self.v_proj = IdentityModule()
        self.o_proj = IdentityModule()
        self.rope = IdentityModule()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.scale = 1.0

    def __call__(self, x, mask=None, cache=None):
        _ = mask
        _ = cache
        return x


def test_attention_wrapper_is_effectively_noop_at_small_scale() -> None:
    genome = KVBiasGenome(hidden_dim=4, kv_dim=4, rank=1, self_dim=2, scale=0.02)
    genome.key_gain = 1e-4
    genome.value_gain = 1e-4

    flat_self = mx.array([1.0 + 1.0j, 1.0 - 1.0j], dtype=mx.complex64)
    fake_attention = FakeAttention()
    wrapper = KVBiasAttention(fake_attention, genome, lambda: flat_self)
    wrapper._attention_fn = lambda q, k, v, cache=None, scale=None, mask=None: v

    x = mx.array([[[0.5, -1.0, 1.5, 2.0]]], dtype=mx.float32)
    wrapped = np.array(wrapper(x))
    baseline = np.array(fake_attention(x))

    np.testing.assert_allclose(wrapped, baseline, rtol=1e-1, atol=3e-2)
