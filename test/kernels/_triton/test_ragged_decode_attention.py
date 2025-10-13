# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.ragged_decode_attention import ragged_decode_attention as ragged_decode_triton

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu",
    reason="Triton tests require GPU backend",
)


def ragged_decode_ref(
    query,
    key,
    value,
    sequence_start,
    sequence_end,
    softmax_scale=1.0,
    block_size=256,
    sliding_window=None,
    logits_soft_cap=None,
    softmax_aux=None,
):
    """
    Dense reference for ragged decode. Matches Triton's streaming behavior:
    - Query at qpos=end-1
    - Valid tokens in [start,end) and within sliding window (if provided)
    - Sinks (aux logits) contribute to normalization mass but not values
    """
    B, HQ, D = query.shape
    S, HKV = key.shape[1], key.shape[2]
    assert HQ % HKV == 0
    G = HQ // HKV

    out = jnp.zeros((B, HQ, D), dtype=query.dtype)
    for b in range(B):
        s_b = int(sequence_start[b])
        e_b = int(sequence_end[b])
        if e_b <= s_b:
            continue
        qpos = e_b - 1

        left = sliding_window[0] if sliding_window is not None else None
        right = sliding_window[1] if sliding_window is not None else None

        for hq in range(HQ):
            kvh = hq // G
            q_vec = query[b, hq]

            t_idx = jnp.arange(S)
            mask = (t_idx >= s_b) & (t_idx < e_b)

            if left is not None:
                mask = mask & (t_idx >= (qpos - left))
            if right is not None:
                mask = mask & (t_idx <= (qpos + right))

            k_b = key[b, :, kvh, :]
            v_b = value[b, :, kvh, :]

            logits = jnp.einsum("d,sd->s", q_vec, k_b) * float(softmax_scale)
            if logits_soft_cap is not None and logits_soft_cap > 0:
                cap = float(logits_soft_cap)
                logits = cap * jnp.tanh(logits / cap)

            logits_seq = jnp.where(mask, logits, -jnp.inf)

            if softmax_aux is None:
                logits_sink = None
            else:
                if softmax_aux.ndim == 2:
                    logits_sink = softmax_aux[kvh]
                else:
                    logits_sink = softmax_aux

                logits_sink = logits_sink.astype(jnp.float32)

            if logits_sink is None:
                m = jnp.max(logits_seq)
                p_seq = jnp.exp(logits_seq - m)
                denom = jnp.sum(p_seq)
            else:
                m = jnp.maximum(jnp.max(logits_seq), jnp.max(logits_sink))
                p_seq = jnp.exp(logits_seq - m)
                p_sink = jnp.exp(logits_sink - m)
                denom = jnp.sum(p_seq) + jnp.sum(p_sink)

            denom = jnp.where(denom == 0.0, 1.0, denom)
            w_seq = p_seq / denom
            o_vec = jnp.einsum("s,sd->d", w_seq, v_b)
            out = out.at[b, hq].set(o_vec.astype(query.dtype))
    return out


class TestRaggedDecodeAttentionTriton:
    def test_forward_shape(self):
        B, S, HQ, HKV, D = 2, 512, 8, 2, 64

        key = jax.random.PRNGKey(0)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([400, 480], dtype=jnp.int32)

        out = ragged_decode_triton(q, k, v, sequence_start=starts, sequence_end=ends, softmax_scale=1.0, block_size=128)
        assert out.shape == (B, HQ, D)
        assert jnp.isfinite(out).all()

    @pytest.mark.parametrize(
        "sliding_window,logits_soft_cap",
        [
            (None, None),
            ((128, 0), None),
            ((256, 64), None),
            (None, 10.0),
            ((128, 64), 20.0),
        ],
    )
    def test_against_reference(self, sliding_window, logits_soft_cap):
        B, S, HQ, HKV, D = 2, 384, 16, 2, 64

        key = jax.random.PRNGKey(1)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 50], dtype=jnp.int32)
        ends = jnp.array([350, 320], dtype=jnp.int32)

        out_tri = ragged_decode_triton(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=1.0,
            block_size=128,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=None,
        )

        out_ref = ragged_decode_ref(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=1.0,
            block_size=128,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=None,
        )

        assert out_tri.shape == out_ref.shape
        assert jnp.allclose(out_tri, out_ref, rtol=1e-4, atol=1e-5)

    def test_with_sinks(self):
        B, S, HQ, HKV, D = 2, 256, 16, 4, 64

        key = jax.random.PRNGKey(2)
        kq, kk, kv, ka = jax.random.split(key, 4)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([10, 0], dtype=jnp.int32)
        ends = jnp.array([200, 220], dtype=jnp.int32)

        NS = 3
        aux_per_kv = jax.random.normal(ka, (HKV, NS), dtype=jnp.float32)
        aux_shared = jnp.linspace(-0.5, 0.5, NS, dtype=jnp.float32)

        out_tri = ragged_decode_triton(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=0.75,
            block_size=64,
            sliding_window=(128, 16),
            logits_soft_cap=25.0,
            softmax_aux=aux_per_kv,
        )
        out_ref = ragged_decode_ref(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=0.75,
            block_size=64,
            sliding_window=(128, 16),
            logits_soft_cap=25.0,
            softmax_aux=aux_per_kv,
        )
        assert jnp.allclose(out_tri, out_ref, rtol=2e-4, atol=2e-5)

        out_tri2 = ragged_decode_triton(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.25,
            block_size=128,
            sliding_window=None,
            logits_soft_cap=None,
            softmax_aux=aux_shared,
        )
        out_ref2 = ragged_decode_ref(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.25,
            block_size=128,
            sliding_window=None,
            logits_soft_cap=None,
            softmax_aux=aux_shared,
        )
        assert jnp.allclose(out_tri2, out_ref2, rtol=2e-4, atol=2e-5)

    def test_tail_block_and_gqa(self):
        B, S, HQ, HKV, D = 1, 410, 32, 4, 64

        key = jax.random.PRNGKey(3)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([5], dtype=jnp.int32)
        ends = jnp.array([407], dtype=jnp.int32)

        out_tri = ragged_decode_triton(q, k, v, starts, ends, softmax_scale=1.0, block_size=128)
        out_ref = ragged_decode_ref(q, k, v, starts, ends, softmax_scale=1.0, block_size=128)
        assert jnp.allclose(out_tri, out_ref, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
