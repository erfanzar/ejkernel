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


"""PagedAttention TPU kernel."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for async copy of multiple K/V pages from HBM."""

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buffer,
        sem,
        block_tables,
        page_indices_start_offset,
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
        self._page_indices = block_tables
        self._page_indices_start_offset = page_indices_start_offset
        self._async_copies = [self._make_async_copy(i) for i in range(self._num_pages_to_load)]

    def _make_async_copy(self, i):
        page_index = self._page_indices[self._page_indices_start_offset + i]
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


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    program_ids=(),
):
    """Pallas kernel for paged attention."""
    if program_ids:
        core_index, b, h, i = program_ids
    else:
        core_index, b, h, i = (
            pl.program_id(0),
            pl.program_id(1),
            pl.program_id(2),
            pl.program_id(3),
        )
    num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
    bk = page_size * pages_per_compute_block
    num_cores = pl.num_programs(0)

    b_step = num_cores if megacore_mode == "batch" else 1
    b_start = core_index if megacore_mode == "batch" else 0
    h_step = num_cores if megacore_mode == "kv_head" else 1
    h_start = core_index if megacore_mode == "kv_head" else 0

    h = h * h_step + h_start
    b = b * b_step + b_start
    length = lengths_ref[b]

    def compute_block_indices(b, h, i):
        def advance_b():
            next_b = b + b_step

            def advance_to_next_non_zero_length():
                next_next_b = next_b + b_step
                return lax.fori_loop(
                    lax.div(next_next_b, b_step),
                    lax.div(batch_size, b_step),
                    lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
                    next_next_b,
                )

            return (
                lax.cond(
                    jnp.logical_and(next_b < batch_size, lengths_ref[lax.clamp(0, next_b, batch_size - 1)] == 0),
                    advance_to_next_non_zero_length,
                    lambda: next_b,
                ),
                h_start,
                0,
            )

        def advance_h():
            next_h = h + h_step
            return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

        return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

    def create_kv_async_copy_descriptors(b, h, i, buffer_index):
        page_offset = b * pages_per_sequence + i * pages_per_compute_block
        pages_to_load = pages_per_compute_block
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            k_sems.at[buffer_index],
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            v_sems.at[buffer_index],
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        return async_copy_k, async_copy_v

    @pl.when(i * bk < length)
    def flash_attention():
        init_flag = init_flag_ref[0]
        init_flag_ref[0] = 0
        buffer_index = buffer_index_ref[0]
        next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

        @pl.when(init_flag)
        def prefetch_first_block():
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(b, h, i, buffer_index)
            async_copy_k.start()
            async_copy_v.start()

        @pl.when(i == 0)
        def init():
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(next_b < batch_size)
        def prefetch_next_block():
            next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
            async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
                next_b, next_h, next_i, next_buffer_index
            )
            async_copy_next_k.start()
            async_copy_next_v.start()
            buffer_index_ref[0] = next_buffer_index

        async_copy_k, async_copy_v = create_kv_async_copy_descriptors(b, h, i, buffer_index)
        q = q_ref[...].astype(jnp.float32)
        k = async_copy_k.wait_and_get_loaded()
        qk = jnp.einsum("gd,td->gt", q, k, preferred_element_type=jnp.float32)
        if attn_logits_soft_cap is not None:
            capped_qk = jnp.tanh(qk / attn_logits_soft_cap)
            qk = capped_qk * attn_logits_soft_cap

        mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
        qk = qk + jnp.where(mask, 0.0, mask_value)
        m_curr = qk.max(axis=-1)

        s_curr = jnp.exp(qk - m_curr[..., None])
        m_prev, l_prev = m_ref[...], l_ref[...]
        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
        m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
        m_next = jnp.maximum(m_prev, m_curr)
        alpha = jnp.exp(m_prev - m_next)
        beta = jnp.exp(m_curr - m_next)
        l_next = alpha * l_prev + beta * l_curr
        m_ref[...], l_ref[...] = m_next, l_next

        v = async_copy_v.wait_and_get_loaded()
        o_curr = jnp.einsum("gt,td->gd", s_curr, v)

        o_ref[...] = ((l_prev * alpha * o_ref[...] + beta * o_curr) / l_next).astype(o_ref.dtype)


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
):
    core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

    def body(i, _):
        paged_flash_attention_kernel(
            lengths_ref,
            page_indices_ref,
            buffer_index_ref,
            init_flag_ref,
            q_ref,
            k_pages_hbm_ref,
            v_pages_hbm_ref,
            o_ref,
            m_ref,
            l_ref,
            k_vmem_buffer,
            v_vmem_buffer,
            k_sems,
            v_sems,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            pages_per_sequence=pages_per_sequence,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            megacore_mode=megacore_mode,
            program_ids=(core_index, b, h, i),
        )
        return ()

    bk = pages_per_compute_block * k_pages_hbm_ref.shape[-2]

    if megacore_mode == "batch":
        num_cores = pl.num_programs(0)
        length = lengths_ref[b * num_cores + core_index]
    else:
        length = lengths_ref[b]

    lax.fori_loop(0, lax.div(length + bk - 1, bk), body, ())
