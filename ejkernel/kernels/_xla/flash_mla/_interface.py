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

"""Flash Multi-head Latent Attention (MLA) interface for XLA backend.

This is a correctness-focused JAX/XLA fallback implementation that matches the
MLA kernel signature used by `ejkernel.modules.operations.flash_mla`.

The core idea is to avoid materializing full K/V by:
  - storing a compressed KV representation (`key_value`)
  - reconstructing per-head keys/values on the fly with `w_kc` / `w_vc`
  - optionally adding a RoPE-derived term via `b_q` / `b_k`
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry


def _repeat_kv_for_gqa(x: Array, q_heads: int) -> Array:
    """Repeat KV heads to match query head count (GQA/MQA support)."""
    kv_heads = int(x.shape[2])
    if kv_heads == q_heads:
        return x
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads}).")
    reps = q_heads // kv_heads
    return jnp.repeat(x, reps, axis=2)


def _flash_mla_xla(
    query: Array,
    key_value: Array,
    w_kc: Array,
    w_vc: Array,
    b_q: Array | None,
    b_k: Array | None,
    softmax_scale: float | None,
    causal: bool,
) -> Array:
    if query.ndim != 4:
        raise ValueError("query must have shape (batch, seq_len, q_heads, head_dim).")
    if key_value.ndim != 3:
        raise ValueError("key_value must have shape (batch, seq_len, kv_lora_rank).")
    if w_kc.ndim != 3 or w_vc.ndim != 3:
        raise ValueError("w_kc/w_vc must have shape (kv_lora_rank, kv_heads, head_dim).")

    batch, seq_len_q, q_heads, q_dim = query.shape
    batch_kv, seq_len_k, kv_lora_rank = key_value.shape
    if batch_kv != batch:
        raise ValueError(f"batch mismatch: query batch={batch} key_value batch={batch_kv}.")
    if seq_len_k != seq_len_q:
        raise ValueError(f"seq_len mismatch: query seq_len={seq_len_q} key_value seq_len={seq_len_k}.")

    if w_kc.shape[0] != kv_lora_rank or w_vc.shape[0] != kv_lora_rank:
        raise ValueError(
            "kv_lora_rank mismatch: "
            f"key_value last dim={kv_lora_rank}, w_kc[0]={w_kc.shape[0]}, w_vc[0]={w_vc.shape[0]}."
        )

    kv_heads = int(w_kc.shape[1])
    if int(w_vc.shape[1]) != kv_heads:
        raise ValueError(f"kv_heads mismatch: w_kc[1]={kv_heads} w_vc[1]={w_vc.shape[1]}.")

    d_nope = int(w_kc.shape[2])
    if b_k is None:
        if q_dim != d_nope:
            raise ValueError(
                "When b_k is None, query head_dim must equal w_kc head_dim. "
                f"Got query head_dim={q_dim}, w_kc head_dim={d_nope}."
            )
    else:
        if b_k.ndim != 3 or b_k.shape[0] != batch or b_k.shape[1] != seq_len_k:
            raise ValueError(
                "b_k must have shape (batch, seq_len, qk_rope_head_dim). "
                f"Got b_k shape={getattr(b_k, 'shape', None)}."
            )
        rope_dim = int(b_k.shape[2])
        expected_q_dim = d_nope + rope_dim
        if b_q is None:
            if q_dim != expected_q_dim:
                raise ValueError(
                    "When b_k is provided and b_q is None, query head_dim must be "
                    f"w_kc head_dim + rope_dim ({expected_q_dim}). Got {q_dim}."
                )
        else:
            if b_q.ndim != 3 or b_q.shape[0] != batch or b_q.shape[1] != seq_len_q:
                raise ValueError(
                    "b_q must have shape (batch, seq_len, qk_rope_head_dim). "
                    f"Got b_q shape={getattr(b_q, 'shape', None)}."
                )
            if int(b_q.shape[2]) != rope_dim:
                raise ValueError(f"b_q/b_k rope_dim mismatch: b_q={b_q.shape[2]} b_k={rope_dim}.")
            if q_dim != d_nope:
                raise ValueError(
                    "When both b_q and b_k are provided, query is expected to contain only the "
                    "non-RoPE part (qk_nope). "
                    f"Got query head_dim={q_dim}, expected {d_nope}."
                )

    q_f32 = query.astype(jnp.float32)
    kv_f32 = key_value.astype(jnp.float32)
    w_kc_f32 = w_kc.astype(jnp.float32)
    w_vc_f32 = w_vc.astype(jnp.float32)

    k_nope = jnp.einsum("btr,rhd->bthd", kv_f32, w_kc_f32, optimize=True)
    v = jnp.einsum("btr,rhd->bthd", kv_f32, w_vc_f32, optimize=True)

    k_nope = _repeat_kv_for_gqa(k_nope, q_heads)
    v = _repeat_kv_for_gqa(v, q_heads)

    if b_k is None:
        logits = jnp.einsum("bqhd,bkhd->bhqk", q_f32, k_nope, optimize=True)
        d_scale = float(d_nope)
    elif b_q is None:
        rope_dim = int(b_k.shape[2])
        q_nope = q_f32[..., :d_nope]
        q_rope = q_f32[..., d_nope : d_nope + rope_dim]
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_nope, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqhd,bkd->bhqk", q_rope, b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope
        d_scale = float(d_nope + rope_dim)
    else:
        logits_nope = jnp.einsum("bqhd,bkhd->bhqk", q_f32, k_nope, optimize=True)
        logits_rope = jnp.einsum("bqd,bkd->bqk", b_q.astype(jnp.float32), b_k.astype(jnp.float32), optimize=True)
        logits = logits_nope + logits_rope[:, None, :, :]
        d_scale = float(d_nope + int(b_k.shape[2]))

    if softmax_scale is None:
        scale = float(1.0 / math.sqrt(d_scale))
    else:
        scale = float(softmax_scale)
    logits = logits * jnp.asarray(scale, dtype=jnp.float32)

    if causal:
        q_idx = jnp.arange(seq_len_q)[:, None]
        k_idx = jnp.arange(seq_len_k)[None, :]
        causal_mask = k_idx <= q_idx
        logits = jnp.where(causal_mask[None, None, :, :], logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v, optimize=True)
    return out.astype(query.dtype)


@kernel_registry.register("flash_mla", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_mla(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Flash MLA (XLA fallback).

    Notes:
        - `cu_seqlens` is currently not supported in this XLA implementation.
        - If `b_k` is provided and `b_q` is None, `query` is expected to contain
          both qk_nope and qk_rope concatenated on the last axis.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens is not supported for XLA flash_mla yet.")
    return _flash_mla_xla(
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
