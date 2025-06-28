# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp

from ejgpu.triton_kernels.flash_attn_varlen._vanilla import attention_vanilla, attention_varlen


def vanilla_attention():
    """vanilla attention function."""
    print("=" * 60)
    print("VANILLA ATTENTION EXAMPLE")
    print("=" * 60)
    keys = jax.random.split(jax.random.PRNGKey(42), 3)
    batch_size = 2
    num_heads = 8
    seq_len = 16
    head_dim = 64
    sm_scale = 1.0 / jnp.sqrt(head_dim)
    q = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim))
    k = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim))
    v = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim))
    print("Input shapes:")
    print(f"  q       : {q.shape}")
    print(f"  k       : {k.shape}")
    print(f"  v       : {v.shape}")
    print(f"  sm_scale: {sm_scale}")
    print("\nRunning attention with causal=False, layout='bhsd'...")
    outputs = attention_vanilla(q, k, v, sm_scale, causal=False, layout="bhsd", use_exp2=False)
    o, softmax_lse, exp_scores, softmax, att_shift_scaled, att_scaled, att_scores = outputs
    print("Output shapes:")
    print(f"  o          : {o.shape}")
    print(f"  softmax_lse: {softmax_lse.shape}")
    print(f"  exp_scores : {exp_scores.shape}")
    print(f"  softmax    : {softmax.shape}")
    print("\nRunning attention with causal=True...")
    outputs_causal = attention_vanilla(q, k, v, sm_scale, causal=True, layout="bhsd", use_exp2=False)
    o_causal = outputs_causal[0]
    print(f"Causal output shape: {o_causal.shape}")
    print(f"Output difference (causal vs non-causal): {jnp.mean(jnp.abs(o - o_causal)):.6f}")
    print("\nTesting with 'bshd' layout...")
    q_bshd = jnp.transpose(q, (0, 2, 1, 3))
    k_bshd = jnp.transpose(k, (0, 2, 1, 3))
    v_bshd = jnp.transpose(v, (0, 2, 1, 3))
    print("Input shapes (bshd):")
    print(f"  q_bshd: {q_bshd.shape}")
    outputs_bshd = attention_vanilla(q_bshd, k_bshd, v_bshd, sm_scale, causal=False, layout="bshd", use_exp2=False)
    o_bshd = outputs_bshd[0]
    print(f"Output shape (bshd): {o_bshd.shape}")
    o_bhsd_from_bshd = jnp.transpose(o_bshd, (0, 2, 1, 3))
    consistency_error = jnp.mean(jnp.abs(o - o_bhsd_from_bshd))
    print(f"Layout consistency error: {consistency_error:.8f}")


def varlen_attention():
    """variable-length attention function."""
    print("\n" + "=" * 60)
    print("VARIABLE LENGTH ATTENTION EXAMPLE")
    print("=" * 60)
    keys = jax.random.split(jax.random.PRNGKey(123), 3)
    batch_size = 3
    num_heads = 4
    head_dim = 32
    sm_scale = 1.0 / jnp.sqrt(head_dim)
    seq_lens_q = [10, 15, 8]
    seq_lens_k = [12, 15, 10]
    total_len_q = sum(seq_lens_q)
    total_len_k = sum(seq_lens_k)
    cu_seqlens_q = jnp.array([0] + [sum(seq_lens_q[: i + 1]) for i in range(batch_size)])
    cu_seqlens_k = jnp.array([0] + [sum(seq_lens_k[: i + 1]) for i in range(batch_size)])
    print("Batch configuration:")
    print(f"  batch_size  : {batch_size}")
    print(f"  num_heads   : {num_heads}")
    print(f"  head_dim    : {head_dim}")
    print(f"  seq_lens_q  : {seq_lens_q}")
    print(f"  seq_lens_k  : {seq_lens_k}")
    print(f"  cu_seqlens_q: {cu_seqlens_q}")
    print(f"  cu_seqlens_k: {cu_seqlens_k}")
    q = jax.random.normal(keys[0], (total_len_q, num_heads, head_dim))
    k = jax.random.normal(keys[1], (total_len_k, num_heads, head_dim))
    v = jax.random.normal(keys[2], (total_len_k, num_heads, head_dim))
    print("\nInput shapes (thd layout):")
    print(f"  q: {q.shape}")
    print(f"  k: {k.shape}")
    print(f"  v: {v.shape}")
    print("\nRunning variable-length attention...")
    outputs = attention_varlen(
        q,
        k,
        v,
        sm_scale,
        False,
        "thd",
        cu_seqlens_q,
        cu_seqlens_k,
        max(seq_lens_q),
        max(seq_lens_k),
        use_exp2=False,
    )
    o, softmax_lse = outputs[0], outputs[1]
    print("Output shapes:")
    print(f"  o          : {o.shape}")
    print(f"  softmax_lse: {softmax_lse.shape}")
    print("\nVerifying individual sequences:")
    for i in range(batch_size):
        start_q = int(cu_seqlens_q[i])
        end_q = int(cu_seqlens_q[i + 1])
        o_i = o[start_q:end_q, :, :]
        lse_i = softmax_lse[start_q:end_q, :]
        print(f"  Sequence {i}: o_shape={o_i.shape}, lse_shape={lse_i.shape}")
        print(f"    Output mean: {jnp.mean(o_i):.6f}, LSE mean: {jnp.mean(lse_i):.6f}")


def comparison_and_benchmarks():
    """Compare different configurations and show performance characteristics."""
    print("\n" + "=" * 60)
    print("COMPARISON AND BENCHMARKS")
    print("=" * 60)
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 3)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 16
    sm_scale = 1.0 / jnp.sqrt(head_dim)
    q = jax.random.normal(keys[0], (batch_size, num_heads, seq_len, head_dim))
    k = jax.random.normal(keys[1], (batch_size, num_heads, seq_len, head_dim))
    v = jax.random.normal(keys[2], (batch_size, num_heads, seq_len, head_dim))
    print(f"Test configuration: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print("\nComparing exp vs exp2:")
    outputs_exp = attention_vanilla(q, k, v, sm_scale, causal=False, layout="bhsd", use_exp2=False)
    outputs_exp2 = attention_vanilla(q, k, v, sm_scale, causal=False, layout="bhsd", use_exp2=True)
    diff = jnp.mean(jnp.abs(outputs_exp[0] - outputs_exp2[0]))
    print(f"  Output difference (exp vs exp2): {diff:.8f}")
    print("\nAttention pattern visualization (first head, first batch):")
    softmax = outputs_exp[3]
    attention_matrix = softmax[0, 0, :, :]
    print(f"Attention matrix shape: {attention_matrix.shape}")
    print("Attention matrix (rounded to 3 decimals):")
    row_sums = jnp.sum(attention_matrix, axis=1)
    print(f"Row sums (should be ~1.0): {jnp.round(row_sums, 6)}")


if __name__ == "__main__":
    vanilla_attention()
    varlen_attention()
    comparison_and_benchmarks()
