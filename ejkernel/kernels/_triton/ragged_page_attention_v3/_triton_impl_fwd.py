import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import triton_call


@triton.jit
def rpa_fa_triton_kernel(
    Q_ptr,
    Kcache_ptr,
    Vcache_ptr,
    BlockTables_ptr,
    kv_lens_ptr,
    q_start_loc_ptr,
    Out_ptr,
    T: tl.constexpr,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    pages_per_seq: tl.constexpr,
    page_size: tl.constexpr,
    num_seqs,
    softmax_scale,
    logits_soft_cap,
    mask_value,
    q_scale,
    k_scale,
    v_scale,
    sliding_window,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    q_blk = tl.program_id(axis=2)

    if seq_id >= num_seqs:
        return

    Hrep = Hq // Hkv
    kv_head = head_id // Hrep

    q_start = tl.load(q_start_loc_ptr + seq_id)
    q_end = tl.load(q_start_loc_ptr + seq_id + 1)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + seq_id)

    kv_cap = pages_per_seq * page_size

    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_len

    tok_m = q_start + offs_m

    rng_n = tl.arange(0, BLOCK_N)

    rng_d = tl.arange(0, D)
    q_row_base = (tok_m[:, None] * Hq + head_id)[..., None] * D + rng_d[None, None]
    q_ptrs = Q_ptr + q_row_base
    Q = tl.where(m_mask[:, None], tl.load(q_ptrs, mask=m_mask[:, None], other=0.0), 0.0)

    if q_scale > 0:
        Q = Q / q_scale

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    o_i = tl.zeros((BLOCK_M, D), tl.float32)

    q_span = (kv_len - q_len) + offs_m

    bt_base = BlockTables_ptr + seq_id * pages_per_seq

    num_k_tiles = (kv_cap + BLOCK_N - 1) // BLOCK_N
    for k_blk in range(0, num_k_tiles):
        offs_n = k_blk * BLOCK_N + rng_n
        n_mask = offs_n < kv_len

        page_id = offs_n // page_size
        in_page = offs_n % page_size
        page_idx = tl.load(bt_base + page_id, mask=(offs_n < kv_cap), other=0)
        row_idx = page_idx * page_size + in_page

        kv_row_base = (row_idx[:, None] * Hkv + kv_head)[..., None] * D + rng_d[None, None]
        k_ptrs = Kcache_ptr + kv_row_base
        v_ptrs = Vcache_ptr + kv_row_base

        K = tl.load(k_ptrs, mask=(offs_n[:, None] < kv_cap), other=0.0)
        V = tl.load(v_ptrs, mask=(offs_n[:, None] < kv_cap), other=0.0)

        K = K.to(tl.float32)
        V = V.to(tl.float32)
        Qf = Q.to(tl.float32)

        S = tl.dot(Qf, tl.trans(K))
        S = S * softmax_scale
        if k_scale > 0:
            S = S * k_scale
        if q_scale > 0:
            S = S * q_scale

        q_ok = m_mask[:, None]
        k_ok = n_mask[None, :]
        causal = q_span[:, None] < offs_n[None, :]
        bad = (~q_ok) | (~k_ok) | causal
        if sliding_window > 0:
            sw = (q_span[:, None] - sliding_window) >= offs_n[None, :]
            bad = bad | sw

        if logits_soft_cap > 0:
            S = logits_soft_cap * tl.tanh(S / logits_soft_cap)
        S = tl.where(bad, S + mask_value, S)

        s_max = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, s_max)
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])

        l_new = alpha * l_i + tl.sum(P, axis=1)
        pv = tl.dot(P, V)

        o_new = alpha[:, None] * o_i + pv

        m_i = m_new
        l_i = l_new
        o_i = o_new

    out_tile = (o_i / l_i[:, None]).to(Q.dtype)

    if v_scale > 0:
        out_tile = out_tile * v_scale

    out_row_base = (tok_m[:, None] * Hq + head_id)[..., None] * D + rng_d[None, None]
    out_ptrs = Out_ptr + out_row_base
    tl.store(out_ptrs, out_tile, mask=m_mask[:, None])


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def fwd_triton_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    block_size: int,
    softmax_scale: float,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
):
    batch_size, seq_len, num_q_heads, head_dim_k = q.shape
    _, _, num_kv_heads, _ = k.shape
    head_dim_v = int(v.shape[-1])
    assert num_q_heads % num_kv_heads == 0, "GQA requires Hq divisible by Hkv"
    num_gqa_groups = num_q_heads // num_kv_heads

    triton_block_size_k = min(128, _next_power_of_2(int(head_dim_k)))
    triton_block_size_v = min(128, _next_power_of_2(int(head_dim_v)))

    num_k_blocks = _cdiv(int(head_dim_k), triton_block_size_k)
    num_v_blocks = _cdiv(int(head_dim_v), triton_block_size_v)
    assert num_k_blocks == 1, "head_dim_k > 128 not supported by this kernel launcher (num_k_blocks must be 1)."

    out_shape = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_q_heads, head_dim_v),
        dtype=v.dtype,
    )
    lse_shape = jax.ShapeDtypeStruct(
        (batch_size, seq_len, num_q_heads),
        dtype=jnp.float32,
    )
    output_shapes = (out_shape, lse_shape)

    kernel_metaparams = dict(
        SEQUENCE=seq_len,
        KV_HEADS=num_kv_heads,
        Q_HEADS=num_q_heads,
        QK_GROUPS=num_gqa_groups,
        BLOCK_DIMK=head_dim_k,
        BLOCK_DIMV=head_dim_v,
        IndicesSize=int(block_indices.shape[-1]),
        BLOCKSIZE=int(block_size),
        BLOCKSIZE_K=triton_block_size_k,
        BLOCKSIZE_V=triton_block_size_v,
    )

    def grid(META):
        return (
            META["SEQUENCE"],
            num_v_blocks,
            batch_size * META["KV_HEADS"],
        )

    block_indices_arg = block_indices if block_indices is not None else jnp.array(1, jnp.int32)
    block_counts_arg = block_counts if block_counts is not None else jnp.array(1, jnp.int32)
    cu_seqlens_arg = cu_seqlens if cu_seqlens is not None else jnp.array(1, jnp.int32)
    token_indices_arg = token_indices if token_indices is not None else jnp.array(1, jnp.int32)

    attn_output, lse = triton_call(
        q,
        k,
        v,
        jnp.asarray(softmax_scale, jnp.float32),
        block_indices_arg,
        block_counts_arg,
        cu_seqlens_arg,
        token_indices_arg,
        kernel=rpa_fa_triton_kernel,
        grid=grid,
        out_shape=output_shapes,
        name="ejkernel::triton::rpa_flash_fwd",
        **kernel_metaparams,
    )

    return attn_output, lse
