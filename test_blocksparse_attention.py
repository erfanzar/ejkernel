#!/usr/bin/env python3
"""Test script for blocksparse attention implementation."""

# Add parent directory to path for imports

import jax
import jax.numpy as jnp
import numpy as np

from ejkernel.kernels._triton.blocksparse_attention import (
    blocksparse_attention,
    generate_blocksparse_layout,
)


def test_blocksparse_attention():
    """Test basic functionality of blocksparse attention."""
    print("Testing Blocksparse Attention Implementation...")

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    block_size = 64
    local_blocks = 4

    print("\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Block size: {block_size}")
    print(f"  Local blocks: {local_blocks}")

    # Generate test data
    q_key, k_key, v_key = jax.random.split(key, 3)
    query = jax.random.normal(q_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    key = jax.random.normal(k_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
    value = jax.random.normal(v_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

    # Generate blocksparse layout
    layout = generate_blocksparse_layout(
        seq_len=seq_len,
        num_heads=1,  # Homogeneous heads
        block_size=block_size,
        local_blocks=local_blocks,
        vert_stride=0,
        homo_head=True,
    )

    num_blocks = seq_len // block_size
    total_blocks = num_blocks * num_blocks
    active_blocks = jnp.sum(layout)
    sparsity = 1.0 - (active_blocks / total_blocks)

    print("\nSparsity pattern:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Active blocks: {active_blocks}")
    print(f"  Sparsity: {sparsity:.2%}")

    try:
        # Test forward pass
        print("\nTesting forward pass...")
        output = blocksparse_attention(
            query=query,
            key=key,
            value=value,
            layout=layout,
            causal=True,
            block_m=block_size,
            block_n=block_size,
        )

        print("✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{jnp.min(output):.4f}, {jnp.max(output):.4f}]")

        # Test gradient computation
        print("\nTesting backward pass...")

        def loss_fn(q, k, v):
            out = blocksparse_attention(
                query=q,
                key=k,
                value=v,
                layout=layout,
                causal=True,
                block_m=block_size,
                block_n=block_size,
            )
            return jnp.mean(out**2)

        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(query, key, value)

        print("✓ Backward pass successful!")
        print(f"  dQ shape: {dq.shape}, range: [{jnp.min(dq):.4f}, {jnp.max(dq):.4f}]")
        print(f"  dK shape: {dk.shape}, range: [{jnp.min(dk):.4f}, {jnp.max(dk):.4f}]")
        print(f"  dV shape: {dv.shape}, range: [{jnp.min(dv):.4f}, {jnp.max(dv):.4f}]")

        # Test with different configurations
        print("\nTesting with vertical stride...")
        layout_with_stride = generate_blocksparse_layout(
            seq_len=seq_len,
            num_heads=1,
            block_size=block_size,
            local_blocks=2,
            vert_stride=4,
            homo_head=True,
        )

        output_stride = blocksparse_attention(
            query=query,
            key=key,
            value=value,
            layout=layout_with_stride,
            causal=True,
            block_m=block_size,
            block_n=block_size,
        )

        active_blocks_stride = jnp.sum(layout_with_stride)
        sparsity_stride = 1.0 - (active_blocks_stride / total_blocks)
        print("✓ Vertical stride test successful!")
        print(f"  Active blocks: {active_blocks_stride}")
        print(f"  Sparsity: {sparsity_stride:.2%}")

        print("\n✅ All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_gradient_correctness():
    """Test gradient correctness with finite differences."""
    print("\nTesting gradient correctness...")

    key = jax.random.PRNGKey(123)

    # Smaller test case for finite difference
    batch_size = 1
    seq_len = 128
    num_heads = 2
    head_dim = 32
    block_size = 32

    # Generate test data
    q_key, k_key, v_key = jax.random.split(key, 3)
    query = jax.random.normal(q_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
    key = jax.random.normal(k_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)
    value = jax.random.normal(v_key, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32)

    # Generate simple layout
    layout = generate_blocksparse_layout(
        seq_len=seq_len,
        num_heads=1,
        block_size=block_size,
        local_blocks=2,
        homo_head=True,
    )

    def loss_fn(q):
        out = blocksparse_attention(
            query=q,
            key=key,
            value=value,
            layout=layout,
            causal=True,
            block_m=block_size,
            block_n=block_size,
        )
        return jnp.sum(out)

    # Compute analytical gradient
    grad_fn = jax.grad(loss_fn)
    grad_analytical = grad_fn(query)

    # Compute finite difference gradient (sample a few elements)
    epsilon = 1e-4
    num_samples = 10
    errors = []

    for _ in range(num_samples):
        # Random position to check
        b = np.random.randint(batch_size)
        s = np.random.randint(seq_len)
        h = np.random.randint(num_heads)
        d = np.random.randint(head_dim)

        # Finite difference
        query_plus = query.at[b, s, h, d].add(epsilon)
        query_minus = query.at[b, s, h, d].add(-epsilon)

        grad_fd = (loss_fn(query_plus) - loss_fn(query_minus)) / (2 * epsilon)
        grad_ana = grad_analytical[b, s, h, d]

        rel_error = np.abs((grad_fd - grad_ana) / (grad_ana + 1e-8))
        errors.append(rel_error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"  Mean relative error: {mean_error:.6f}")
    print(f"  Max relative error: {max_error:.6f}")

    if max_error < 0.01:  # 1% tolerance
        print("✓ Gradient correctness test passed!")
        return True
    else:
        print("❌ Gradient correctness test failed!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("BLOCKSPARSE ATTENTION TEST SUITE")
    print("=" * 60)

    success = test_blocksparse_attention()
    if success:
        test_gradient_correctness()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
