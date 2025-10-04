# ejKernel High-Performance JAX Kernels for Deep Learning from EasyDeL

> *"The best optimization is the one you don't have to think about."*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7.2+-orange.svg)](https://github.com/google/jax)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-green.svg)](https://ejkernel.readthedocs.io/en/latest/)

ejKernel (EasyDeL JAX Kernels) is a high-performance library providing optimized kernel implementations for deep learning workloads in JAX. It offers multi-platform support with automatic backend selection, enabling efficient execution across GPUs, TPUs, and CPUs through Triton, Pallas, CUDA, and XLA backends.

## Key Features

### Multi-Platform Kernel Registry

- **Automatic Platform Detection**: Seamlessly selects optimal implementation based on hardware
- **Backend Support**: GPU (CUDA/Triton), TPU (Pallas/XLA), CPU (XLA)
- **Priority-based Selection**: Configurable kernel selection with fallback mechanisms

### State-of-the-Art Attention Mechanisms

- **Flash Attention**: Memory-efficient O(N) attention with causal masking, dropout, and sliding windows
- **Page Attention**: Optimized for KV-cache in inference scenarios
- **Ring Attention**: Distributed attention for sequence parallelism
- **Sparse Attention**: Block-sparse patterns for efficient long-context processing
- **GLA (Gated Linear Attention)**: Linear complexity attention alternative
- **Lightning Attention**: Layer-dependent decay attention mechanism
- **MLA (Multi-head Latent Attention)**: Efficient latent attention implementation

### Advanced Operations

- **Recurrent Kernels**: Optimized RNN-like operations with custom gradients
- **Mean Pooling**: Variable-length sequence pooling with proper masking
- **Grouped Matrix Multiplication**: Efficient batched matrix operations
- **Native Sparse Operations**: Block-sparse matrix computations

### Developer-Friendly Design

- **Type Hints**: Full jaxtyping annotations for better IDE support
- **Modular Architecture**: Easy to extend with new kernel implementations
- **Comprehensive Testing**: Extensive test coverage with XLA vs Triton comparisons
- **Automatic Differentiation**: Custom VJP rules for efficient gradients

## Installation

### Basic Installation

```bash
pip install ejkernel
```

### GPU Support (CUDA)

```bash
pip install ejkernel[gpu]
```

### TPU Support

```bash
pip install ejkernel[tpu]
```

### Development Installation

```bash
git clone https://github.com/erfanzar/ejkernel.git
cd ejkernel
pip install -e .
```

## Quick Start

### Basic Flash Attention

```python
import jax
import jax.numpy as jnp
from ejkernel.modules import FlashAttention, create_default_executor

# Initialize
executor = create_default_executor()
attention = FlashAttention()

# Create inputs
batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
key = jax.random.PRNGKey(0)
q = k = v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

# Execute attention
output = executor(attention, q, k, v, causal=True, dropout_prob=0.1)
```

### Using Kernel Registry Directly

```python
from ejkernel import kernel_registry, Platform, Backend

# Get specific implementation
flash_attn_gpu = kernel_registry.get(
    algorithm="flash_attention",
    platform=Platform.TRITON,
    backend=Backend.GPU
)

# Execute kernel
output = flash_attn_gpu(query, key, value, causal=True)
```

### Page Attention for Inference

```python
from ejkernel.kernels._xla.page_attention import page_attention

# Setup KV cache
max_blocks, block_size = 256, 16
key_cache = jnp.zeros((num_blocks, num_heads, block_size, head_dim))
value_cache = jnp.zeros((num_blocks, num_heads, block_size, head_dim))

# Block tables for sequence mapping
block_tables = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])
context_lens = jnp.array([48, 32])

# Run paged attention
output = page_attention(
    query, key_cache, value_cache,
    context_lens, block_tables,
    scale=0.125, block_size=block_size
)
```

## Architecture

### Project Structure

```md
ejkernel/
├── kernels/           # Core kernel implementations
│   ├── _triton/      # Triton GPU kernels
│   ├── _pallas/      # Pallas TPU kernels
│   ├── _xla/         # XLA CPU/fallback kernels
│   └── _cuda/        # Native CUDA kernels
├── modules/          # High-level operation modules
│   └── operations/   # Wrapped kernels with auto-selection
├── ops/              # Kernel execution framework
└── utils.py          # Utilities and helpers
```

### Kernel Registry System

The kernel registry enables automatic platform selection:

```python
@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU, priority=100)
def my_kernel_triton(x, y):
    # Triton implementation
    ...

@kernel_registry.register("my_kernel", Platform.XLA, Backend.ANY, priority=50)
def my_kernel_xla(x, y):
    # XLA fallback implementation
    ...

# Automatic selection based on hardware
impl = kernel_registry.get("my_kernel")  # Selects best available
```

## Supported Algorithms

### Currently Implemented

#### Attention Mechanisms

- **Flash Attention v2**: Memory-efficient exact attention with O(N) memory complexity
  - Causal masking, dropout, sliding windows
  - Variable-length sequence support (cu_seqlens)
  - Logits soft capping (Gemma-style)
  - Multi-query (MQA) and grouped-query (GQA) attention
- **Page Attention**: Optimized for inference with paged KV-cache
  - Block-wise memory management
  - Dynamic context lengths
  - Support for continuous batching
- **Ring Attention**: Distributed attention for sequence parallelism
  - Enables training on ultra-long sequences
  - Communication-computation overlap
- **Native Sparse Attention**: Block-sparse patterns for efficiency
  - Configurable sparsity patterns
  - Local + global attention combinations
- **GLA (Gated Linear Attention)**: Linear complexity O(N) attention
  - Gated recurrent updates
  - Efficient for very long sequences
- **Lightning Attention**: Linear attention with layer-wise decay
  - Exponential moving average mechanism
  - Improved long-range modeling
- **MLA (Multi-head Latent Attention)**: Compressed KV representation
  - Reduces memory footprint
  - Maintains attention quality
- **Ragged Page Attention**: Variable-length paged attention
  - Handles sequences of different lengths efficiently
  - Optimized for batched inference

#### Pooling & Aggregation

- **Mean Pooling**: Sequence-level representations
  - Support for variable-length sequences
  - Proper masking and normalization
  - Custom gradients for efficiency

#### Linear Algebra

- **Grouped Matrix Multiplication**: Batched GEMM operations
  - Efficient for multi-head computations
  - Optimized memory access patterns

#### Recurrent Operations

- **Recurrent Kernels**: RNN-like sequential processing
  - Custom backward pass
  - Hidden state caching
  - Support for bidirectional processing

### Platform Support Matrix

| Algorithm | Triton GPU | Pallas TPU | XLA (CPU/Fallback) | CUDA |
|-----------|------------|------------|--------------------|------|
| Flash Attention v2 | ✅ | ✅ | ✅ | 🚧 |
| Page Attention | ✅ | ✅ | ✅ | 🚧 |
| Ring Attention | ✅ | ✅ | ✅ | 🚧 |
| Native Sparse | ✅ | ❌ | ✅ | 🚧 |
| GLA | ✅ | 🚧 | ✅ | ❌ |
| Lightning Attention | ✅ | ❌ | ✅ | 🚧 |
| MLA | ✅ | 🚧 | ❌ | ❌ |
| Ragged Page Attention | ✅ | ✅ | ✅ | 🚧 |
| Recurrent | ✅ | 🚧 | ✅ | 🚧 |
| Mean Pooling | ✅ | 🚧 | ✅ | 🚧 |
| Grouped MatMul | 🚧 | ✅ | ✅ | 🚧 |

✅ = Implemented and optimized
🚧 = Under development
❌ = Not yet implemented

### Coming Soon

We're actively working on expanding our algorithm support. Upcoming implementations include:

#### Attention Variants

- **Flash Attention 3**: Next-generation with further optimizations
- **Flash Decoding**: Optimized for inference parallelism
- **Sliding Window Attention with Sinks**: Attention sinks for streaming
- **Differential Attention**: Learnable attention patterns
- **Mixture of Attention**: Dynamic attention mechanism selection
- **Alibi/RoPE/XPos**: Position encoding variants
- **RetNet**: Retention-based architecture
- **RWKV Attention**: Linear complexity with RNN-like properties
- **Linformer**: Low-rank factorization for efficiency
- **Performer**: FAVOR+ algorithm with kernel approximation

#### Optimization Techniques

- **Speculative Decoding Kernels**: Accelerated inference
- **Continuous Batching**: Dynamic batch management
- **Quantized Attention**: INT8/INT4 operations
- **Sparse Flash Attention**: Combining sparsity with flash attention
- **Cross-Attention Optimization**: Encoder-decoder specific kernels

#### Advanced Non-Attn Operations

- **Fused LayerNorm + Attention**: Reduced memory transfers
- **Fused MLP Kernels**: Optimized feed-forward networks
- **RMSNorm Kernels**: Efficient normalization
- **Rotary Embeddings**: Optimized RoPE implementation
- **Mamba SSM Kernels**: State-space model operations
- **Dynamic Sparse Training**: Learnable sparsity patterns

#### Memory Management

- **Gradient Checkpointing Kernels**: Memory-efficient training
- **Activation Compression**: On-the-fly compression/decompression
- **Memory-Efficient Backward**: Reduced activation storage

### Contributing New Algorithms

We welcome contributions of new algorithms! If you'd like to add support for a new operation:

1. Check our [contribution guidelines](CONTRIBUTING.md)
2. Implement the kernel in the appropriate backend directory
3. Add comprehensive tests
4. Submit a pull request

Priority areas for contribution:

- TPU/Pallas implementations for existing algorithms
- CUDA native kernels for maximum performance
- New attention mechanisms from recent papers
- Fusion opportunities for common patterns

## Testing

### Run All Tests

```bash
python test/run_tests.py
```

### Platform-Specific Tests

```bash
# XLA implementation tests
python test/run_tests.py --xla

# Triton implementation tests
python test/run_tests.py --triton

# Comparison tests (XLA vs Triton)
python test/run_tests.py --comparison
```

### Specific Test Patterns

```bash
python test/run_tests.py -k "flash_attention"
python test/run_tests.py --verbose --failfast
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/erfanzar/ejkernel.git
cd ejkernel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Adding New Kernels

1. Implement kernel in appropriate backend directory
2. Register with `@kernel_registry.register` decorator
3. Add tests in `test/kernels/`
4. Update documentation

## Documentation

Full documentation is available at [ejkernel.readthedocs.io](https://ejkernel.readthedocs.io/en/latest/)

### API Reference

- [Kernel Registry](https://ejkernel.readthedocs.io/en/latest/api/registry.html)
- [Attention Modules](https://ejkernel.readthedocs.io/en/latest/api/attention.html)
- [Operations](https://ejkernel.readthedocs.io/en/latest/api/operations.html)

### Tutorials

- [Getting Started with ejKernel](https://ejkernel.readthedocs.io/en/latest/tutorials/getting_started.html)
- [Custom Kernel Development](https://ejkernel.readthedocs.io/en/latest/tutorials/custom_kernels.html)
- [Performance Optimization](https://ejkernel.readthedocs.io/en/latest/tutorials/optimization.html)

## Citation

If you use ejKernel in your research, please cite:

```bibtex
@software{ejkernel2024,
  author = {Erfan Zare Chavoshi},
  title = {ejKernel: High-Performance JAX Kernels for Deep Learning},
  year = {2024},
  url = {https://github.com/erfanzar/ejkernel}
}
```

## License

ejKernel is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

ejKernel builds upon several excellent projects:

- [JAX](https://github.com/google/jax) - Composable transformations of Python+NumPy programs
- [Triton](https://github.com/openai/triton) - Language and compiler for GPU programming
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Fast and memory-efficient attention
- [EasyDeL](https://github.com/erfanzar/EasyDeL) - Parent framework for JAX deep learning

## Community

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/erfanzar/ejkernel/issues)
- **Discussions**: [Community discussions](https://github.com/erfanzar/ejkernel/discussions)
- **Email**: <Erfanzare810@gmail.com>

---

*ejKernel</b> - Accelerating JAX with optimized kernels*
