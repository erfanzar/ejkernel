EjGPU 🔮
==========

.. _EjGPU:

EjGPU provides GPU Kernels implemented in Triton, primarily designed for use within the `EasyDeL`_ library. It offers highly optimized implementations of various operations, with a focus on attention mechanisms, to accelerate deep learning model training and inference on GPUs.

.. _EasyDeL: https://github.com/erfanzar/EasyDeL

Features
--------

* **Triton Kernels:** Optimized GPU kernels for various operations, leveraging the Triton language for high performance.
* **Integration with EasyDeL:** Designed to work seamlessly with the EasyDeL library, providing accelerated components for large-scale language models.
* **Attention Mechanisms:** Includes highly optimized implementations for:
    * Flash Attention (variable length)
    * Recurrent Attention
    * Lightning Attention
    * Native Sparse Attention
* **Utility Functions:** Provides helper functions for colored and lazy logging, array manipulation, and retrieving device information.
* **XLA Utilities:** Includes utilities for handling packed sequences, calculating cumulative sums, and managing chunking for efficient processing on XLA devices.

Installation
------------

(Assuming standard Python package installation)

.. code-block:: bash

   pip install ejgpu

Usage
-----

Here are a few examples demonstrating how to use some of the kernels provided by ejgpu:

Flash Attention (Variable Length)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from ejgpu import flash_attn_varlen
   from ejgpu.utils import varlen_input_helper

   # Example usage of variable length flash attention
   batch_size = 4
   qheads = 8
   kvheads = 8
   qseqlen = 512
   kseqlen = 512
   head_dim = 64

   # Prepare variable length inputs
   q, k, v, metadata = varlen_input_helper(
       batch_size=batch_size,
       qheads=qheads,
       kvheads=kvheads,
       qseqlen=qseqlen,
       kseqlen=kseqlen,
       head_dim=head_dim,
       dtype=jnp.float16,
       equal_seqlens=False, # Set to True for equal length sequences
   )

   # Apply flash attention
   output = flash_attn_varlen(
       q,
       k,
       v,
       cu_seqlens_q=metadata.cu_seqlens_q,
       cu_seqlens_k=metadata.cu_seqlens_k,
       max_seqlens_q=metadata.max_seqlens_q,
       max_seqlens_k=metadata.max_seqlens_k,
       causal=True, # Set to False for non-causal attention
       layout=metadata.layout,
       sm_scale=metadata.sm_scale,
   )

   print("Flash Attention (Variable Length) Output Shape:", output.shape)

Recurrent Attention
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from ejgpu import recurrent
   from ejgpu.utils import numeric_gen

   # Example usage of recurrent attention
   batch_size = 2
   seq_len = 256
   num_heads = 4
   key_dim = 64
   value_dim = 64

   query = numeric_gen(batch_size, seq_len, num_heads, key_dim)
   key = numeric_gen(batch_size, seq_len, num_heads, key_dim)
   value = numeric_gen(batch_size, seq_len, num_heads, value_dim)
   init_state = numeric_gen(batch_size, num_heads, key_dim, value_dim)

   # Apply recurrent attention
   output, state = recurrent(query, key, value, initial_state=init_state)

   print("Recurrent Attention Output Shape:", output.shape)
   print("Recurrent Attention Final State Shape:", state.shape)

Native Sparse Attention
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from ejgpu import native_spare_attention
   from ejgpu.utils import generate_block_indices, numeric_gen

   # Example usage of native sparse attention
   batch_size = 1
   kv_heads = 1
   query_heads = 16
   sequence_length = 128
   head_dim = 64
   num_blocks_per_token = 8
   block_size = 16
   scale = 0.1

   query = numeric_gen(batch_size, sequence_length, query_heads, head_dim)
   key = numeric_gen(batch_size, sequence_length, kv_heads, head_dim)
   value = numeric_gen(batch_size, sequence_length, kv_heads, head_dim)
   block_indices = generate_block_indices(
       batch_size,
       sequence_length,
       kv_heads,
       num_blocks_per_token,
       block_size
   )

   # Apply native sparse attention
   output = native_spare_attention(
       q=query,
       k=key,
       v=value,
       block_indices=block_indices,
       block_size=block_size,
       scale=scale
   )

   print("Native Sparse Attention Output Shape:", output.shape)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   api_docs/apis
