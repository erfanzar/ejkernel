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

# Reference: JetStream chunked prefill attention
# https://github.com/AI-Hypercomputer/JetStream/blob/main/experimental/jax/inference/kernel/attention/tpu/chunked_prefill_attention.py

"""Chunked Prefill PagedAttention TPU kernel."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def ref_prefill_page_attention(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    context_len: jax.Array,
    page_indices: jax.Array,
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
) -> jax.Array:
    """Reference implementation of chunked prefill paged attention for testing.

    This processes a single sequence's prefill with paged KV cache.

    Args:
        query: A [chunk_size, num_q_heads, head_dim] jax.Array.
        key_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        value_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        context_len: Scalar or [1] jax.Array - the total context length (including this chunk).
        page_indices: A [num_pages_needed] jax.Array of page indices for this sequence.
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim)).
        mask_value: The value used for padding/causal masking in attention.
        attn_logits_soft_cap: The value used for soft capping the attention logits.
        sliding_window: If set, only attend to the last `sliding_window` tokens.

    Returns:
        The output of attention [chunk_size, num_q_heads, head_dim].
    """
    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, _ = key_cache.shape
    num_groups = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    # Get context length as scalar
    length = int(context_len.reshape(-1)[0]) if hasattr(context_len, "reshape") else int(context_len)

    # Calculate number of pages needed
    num_pages = (length + page_size - 1) // page_size

    # Gather K/V from paged cache
    k_pages = key_cache[:, page_indices[:num_pages]]  # [num_kv_heads, num_pages, page_size, head_dim]
    v_pages = value_cache[:, page_indices[:num_pages]]  # [num_kv_heads, num_pages, page_size, head_dim]

    # Reshape to [num_kv_heads, seq_len, head_dim]
    k = k_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]
    v = v_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]

    # Repeat K/V for grouped query attention
    k = jnp.repeat(k, num_groups, axis=0)  # [num_q_heads, seq_len, head_dim]
    v = jnp.repeat(v, num_groups, axis=0)  # [num_q_heads, seq_len, head_dim]

    # Compute attention scores: [chunk_size, num_q_heads, seq_len]
    # q: [chunk_size, num_q_heads, head_dim]
    # k: [num_q_heads, seq_len, head_dim]
    qk = jnp.einsum("qhd,hsd->hqs", query, k, preferred_element_type=jnp.float32)
    qk = qk * softmax_scale

    # Apply soft cap if specified
    if attn_logits_soft_cap is not None:
        qk = attn_logits_soft_cap * jnp.tanh(qk / attn_logits_soft_cap)

    # Create causal mask
    # Query positions: last chunk_size positions of the sequence
    # q_pos[i] = length - chunk_size + i
    q_positions = (length - chunk_size) + jnp.arange(chunk_size)  # [chunk_size]
    kv_positions = jnp.arange(length)  # [seq_len]

    # Causal: q_pos >= kv_pos (can attend to current and past)
    causal_mask = q_positions[:, None] >= kv_positions[None, :]  # [chunk_size, seq_len]

    # Apply sliding window if specified
    if sliding_window is not None:
        # Can only attend to kv_pos >= q_pos - sliding_window + 1
        sliding_mask = kv_positions[None, :] >= (q_positions[:, None] - sliding_window + 1)
        causal_mask = jnp.logical_and(causal_mask, sliding_mask)

    # Broadcast mask to [num_q_heads, chunk_size, seq_len]
    causal_mask = jnp.broadcast_to(causal_mask, (num_q_heads, chunk_size, length))

    qk = qk + jnp.where(causal_mask, 0.0, mask_value)

    # Softmax
    attn = jax.nn.softmax(qk, axis=-1)

    # Compute output: [num_q_heads, chunk_size, head_dim]
    out = jnp.einsum("hqs,hsd->qhd", attn, v)

    return out.astype(query.dtype)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for async copy of multiple K/V pages from HBM."""

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buffer,
        sem,
        page_indices,
        page_offset,
        num_pages_to_load,
        head_index,
    ):
        self._vmem_buffer = vmem_buffer
        self._num_pages_to_load = num_pages_to_load
        if head_index is not None:
            self._pages_hbm_ref = pages_hbm_ref.at[head_index]
        else:
            self._pages_hbm_ref = pages_hbm_ref
        self._sem = sem
        self._page_indices = page_indices
        self._page_offset = page_offset
        self._async_copies = [self._make_async_copy(i) for i in range(self._num_pages_to_load)]

    def _make_async_copy(self, i):
        page_index = self._page_indices[self._page_offset + i]
        return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem)

    def start(self):
        """Starts the async copies."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait_and_get_loaded(self) -> jax.Array:
        """Wait async copies and gets the loaded buffer as a jax.Array."""
        for async_copy in self._async_copies:
            async_copy.wait()
        head_dim = self._vmem_buffer.shape[-1]
        jax_array = self._vmem_buffer[...].astype(jnp.float32)
        return jax_array.reshape(-1, head_dim)


def chunked_prefill_attention_kernel(
    length_ref,
    page_indices_ref,
    buffer_index_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    out_ref,
    l_ref,
    m_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    chunk_size: int,
    page_size: int,
    num_kv_chunks: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    sliding_window: int | None,
):
    """Pallas kernel for chunked prefill attention with paged KV cache.

    This kernel processes the last `chunk_size` tokens of a sequence during prefill.
    The query positions are: [length - chunk_size, length - chunk_size + 1, ..., length - 1]
    Each query position can attend to all KV positions from 0 to its own position (causal).
    """
    h = pl.program_id(0)
    head_dim = k_pages_hbm_ref.shape[3]
    # q_ref shape is [1, group_size, chunk_size, head_dim]
    group_size = q_ref.shape[1]
    length = length_ref[0]

    # Initialize accumulators
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    out_ref[...] = jnp.zeros_like(out_ref)

    pages_per_chunk = chunk_size // page_size

    # The query chunk starts at position (length - chunk_size)
    # So query position q (0 to chunk_size-1) maps to absolute position (length - chunk_size + q)
    q_start_pos = length - chunk_size

    def create_kv_async_copy_descriptors(i, buffer_index):
        """Create async copy descriptors for KV chunk i."""
        page_offset = i * pages_per_chunk
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_per_chunk,
            head_index=h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_per_chunk,
            head_index=h,
        )
        return async_copy_k, async_copy_v

    def per_kv_chunk_body(i, _):
        """Process KV chunk i."""

        @pl.when((i * chunk_size) < length)
        def body():
            buffer_index = buffer_index_ref[0]

            # Prefetch first block on first iteration for head 0
            @pl.when((i == 0) & (h == 0))
            def prefetch_first_kv():
                async_copy_k, async_copy_v = create_kv_async_copy_descriptors(0, buffer_index)
                async_copy_k.start()
                async_copy_v.start()

            # Prefetch next block if there is one
            next_i = i + 1

            @pl.when((next_i < num_kv_chunks) & (next_i * chunk_size < length))
            def prefetch_next_block():
                next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
                async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(next_i, next_buffer_index)
                async_copy_next_k.start()
                async_copy_next_v.start()
                buffer_index_ref[0] = next_buffer_index

            # Load current KV chunk
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(i, buffer_index)
            k = async_copy_k.wait_and_get_loaded()
            v = async_copy_v.wait_and_get_loaded()

            # Create causal mask
            # Query positions: q_start_pos + [0, 1, ..., chunk_size-1]
            # KV positions for chunk i: i * chunk_size + [0, 1, ..., chunk_size-1]
            mask_shape = (chunk_size, chunk_size)
            q_positions = q_start_pos + lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            kv_positions = i * chunk_size + lax.broadcasted_iota(jnp.int32, mask_shape, 1)

            # Causal mask: q_position >= kv_position
            causal_mask = q_positions >= kv_positions

            # Also mask out KV positions beyond the sequence length
            valid_kv_mask = kv_positions < length
            causal_mask = jnp.logical_and(causal_mask, valid_kv_mask)

            # Apply sliding window mask if specified
            if sliding_window is not None:
                # kv_position >= q_position - sliding_window + 1
                sliding_mask = kv_positions >= (q_positions - sliding_window + 1)
                causal_mask = jnp.logical_and(causal_mask, sliding_mask)

            causal_mask_value = jnp.where(causal_mask, 0.0, mask_value)

            def per_group_body(group_idx, _):
                # q_ref shape is [1, group_size, chunk_size, head_dim]
                q = q_ref[0, group_idx]  # [chunk_size, head_dim]
                s = jnp.einsum("td,sd->ts", q, k, preferred_element_type=jnp.float32) + causal_mask_value

                if attn_logits_soft_cap is not None:
                    s = attn_logits_soft_cap * jnp.tanh(s / attn_logits_soft_cap)

                s_max = jnp.max(s, axis=1, keepdims=True)

                prev_m = m_ref[0, group_idx]
                prev_l = l_ref[0, group_idx]

                cur_m = jnp.maximum(prev_m, s_max)
                cur_m_to_attn_size = lax.broadcast_in_dim(cur_m, (chunk_size, chunk_size), (0, 1))

                p = jnp.exp(s - cur_m_to_attn_size)

                cur_l = jnp.exp(prev_m - cur_m) * prev_l + jnp.sum(p, axis=1, keepdims=True)

                out = out_ref[0, group_idx]

                out_ref[0, group_idx, :, :] = (
                    out * lax.broadcast_in_dim(jnp.exp(prev_m - cur_m), (chunk_size, head_dim), (0, 1)) + p @ v
                ).astype(out_ref.dtype)

                m_ref[0, group_idx, :, :] = cur_m
                l_ref[0, group_idx, :, :] = cur_l
                return ()

            lax.fori_loop(0, group_size, per_group_body, ())

        return ()

    # Use static num_kv_chunks instead of dynamic computation
    lax.fori_loop(0, num_kv_chunks, per_kv_chunk_body, ())

    # Final rescaling by l
    # l_ref shape is [1, group_size, chunk_size, 1]
    l = lax.broadcast_in_dim(l_ref[...], (1, group_size, chunk_size, head_dim), (0, 1, 2, 3))
    out_ref[...] = (out_ref[...] / l).astype(out_ref.dtype)
