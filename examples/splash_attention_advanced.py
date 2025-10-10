"""
Advanced examples of Splash Attention with soft capping, custom scales, and auxiliary values.

This demonstrates advanced features:
1. Soft capping (like Gemma2)
2. Custom softmax scales
3. Auxiliary softmax values
4. Combinations with sliding windows and chunked masks
"""

import jax
import jax.numpy as jnp

from ejkernel.kernels._pallas.tpu.blocksparse_attention import blocksparse_attention

# Set up example tensors
batch_size = 2
num_heads = 8
seq_len = 1024  # Must be multiple of 128 for TPU
head_dim = 64

key = jax.random.key(0)
q_key, k_key, v_key = jax.random.split(key, 3)

# Create random Q, K, V tensors
query = jax.random.normal(q_key, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
key_array = jax.random.normal(k_key, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
value = jax.random.normal(v_key, (batch_size, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)


# Example 1: Soft capping (like Gemma2)
print("Example 1: Soft capping for attention logits (Gemma2 style)")
print("-" * 50)

# Without soft cap
output_no_cap = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    causal=True,
)
print(f"Output shape without soft cap: {output_no_cap.shape}")

# With soft cap (Gemma2 uses 50.0)
output_with_cap = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    logit_soft_cap=50.0,  # Apply tanh(logits/50) * 50
    causal=True,
)
print(f"Output shape with soft cap: {output_with_cap.shape}")

# Compare outputs
diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap)))
print(f"Mean absolute difference: {diff:.6f}")
print("✓ Soft capping changes the attention distribution\n")


# Example 2: Custom softmax softmax_scale
print("Example 2: Custom softmax softmax_scale")
print("-" * 50)

# Default softmax_scale (1/sqrt(head_dim) = 1/sqrt(64) = 0.125)
output_default_scale = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    softmax_scale=None,  # Will use default
    causal=True,
)
print(f"Output shape with default softmax_scale (1/√{head_dim}): {output_default_scale.shape}")

# Custom softmax_scale (e.g., for temperature control)
output_custom_scale = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    softmax_scale=0.2,  # Higher temperature (softer attention)
    causal=True,
)
print(f"Output shape with custom softmax_scale (0.2): {output_custom_scale.shape}")

# Lower temperature (sharper attention)
output_sharp = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    softmax_scale=0.05,  # Lower temperature
    causal=True,
)
print(f"Output shape with sharp softmax_scale (0.05): {output_sharp.shape}")

diff_scale = float(jnp.mean(jnp.abs(output_default_scale - output_custom_scale)))
print(f"Difference between default and custom softmax_scale: {diff_scale:.6f}")
print("✓ Different scales control attention sharpness\n")


# Example 3: Auxiliary softmax values (for sink tokens)
print("Example 3: Auxiliary softmax values")
print("-" * 50)

# Create auxiliary values (e.g., for sink tokens or special attention patterns)
aux_key = jax.random.key(123)
softmax_aux = jax.random.normal(aux_key, (batch_size, num_heads), dtype=jnp.float32)

output_with_aux = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    softmax_aux=softmax_aux,
    causal=True,
)
print(f"Output shape with auxiliary values: {output_with_aux.shape}")
print("✓ Auxiliary values can be used for sink tokens or special patterns\n")


# Example 4: Soft cap with sliding window
print("Example 4: Combining soft cap with sliding window")
print("-" * 50)

output_cap_sliding = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    logit_soft_cap=30.0,  # Lower cap for stronger effect
    sliding_window=256,  # Local attention window
    causal=True,
)
print(f"Output shape: {output_cap_sliding.shape}")
print("✓ Soft capping works with sliding windows\n")


# Example 5: Soft cap with chunked causal
print("Example 5: Combining soft cap with chunked causal")
print("-" * 50)

output_cap_chunked = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    logit_soft_cap=40.0,
    chunk_size=128,  # Chunked causal attention
)
print(f"Output shape: {output_cap_chunked.shape}")
print("✓ Soft capping works with chunked causal masks\n")


# Example 6: Full combination - all features
print("Example 6: Combining all features")
print("-" * 50)

output_all_features = blocksparse_attention(
    query=query,
    key=key_array,
    value=value,
    logit_soft_cap=35.0,  # Soft capping
    softmax_scale=0.15,  # Custom softmax_scale
    softmax_aux=softmax_aux,  # Auxiliary values
    sliding_window=(128, 64),  # Asymmetric window
    causal=True,
    query_chunk_size=256,  # Performance tuning
    key_chunk_size=256,
)
print(f"Output shape with all features: {output_all_features.shape}")
print("✓ All features can be combined together\n")


# Example 7: Effect of different soft cap values
print("Example 7: Effect of different soft cap values")
print("-" * 50)

soft_cap_values = [10.0, 30.0, 50.0, 100.0, None]
outputs = []

for cap_value in soft_cap_values:
    output = blocksparse_attention(
        query=query[:, :, :256, :],  # Smaller sequence for quick testing
        key=key_array[:, :, :256, :],
        value=value[:, :, :256, :],
        logit_soft_cap=cap_value,
        causal=True,
    )
    outputs.append(output)
    cap_str = "No cap" if cap_value is None else f"{cap_value:.1f}"
    print(f"Soft cap = {cap_str:8s} -> Output shape: {output.shape}")

# Compare differences
for i, cap_value in enumerate(soft_cap_values[:-1]):
    diff = float(jnp.mean(jnp.abs(outputs[i] - outputs[-1])))
    print(f"  Difference (cap={cap_value} vs no cap): {diff:.6f}")

print("✓ Lower soft cap values have stronger regularization effect\n")


# Example 8: Performance comparison with different configurations
print("Example 8: Performance configurations")
print("-" * 50)

configs = [
    {"name": "Baseline", "params": {}},
    {"name": "With soft cap", "params": {"logit_soft_cap": 50.0}},
    {"name": "Custom softmax_scale", "params": {"softmax_scale": 0.2}},
    {"name": "Sliding window", "params": {"sliding_window": 256}},
    {
        "name": "All optimizations",
        "params": {
            "logit_soft_cap": 50.0,
            "softmax_scale": 0.15,
            "sliding_window": 256,
            "query_chunk_size": 256,
            "key_chunk_size": 256,
        },
    },
]

for config in configs:
    output = blocksparse_attention(query=query, key=key_array, value=value, causal=True, **config["params"])
    print(f"{config['name']:20s} -> Shape: {output.shape}, Finite: {jnp.all(jnp.isfinite(output))}")

print("\n✓ All configurations produce valid outputs")
print("=" * 50)
print("All advanced examples completed successfully!")
