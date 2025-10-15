# ejKernel: High-Performance JAX Kernels for Deep Learning

> *"The best optimization is the one you don't have to think about."*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7.2+-orange.svg)](https://github.com/google/jax)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-green.svg)](https://ejkernel.readthedocs.io/en/latest/)

ejKernel is a production-grade kernel library for JAX that provides highly optimized implementations of deep learning operations with automatic multi-backend support. The library features a sophisticated configuration management system with autotuning, comprehensive type safety, and seamless execution across GPUs, TPUs, and CPUs.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Supported Operations](#supported-operations)
- [Advanced Usage](#advanced-usage)
- [Performance](#performance)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Key Features

### Intelligent Kernel Management

- **7-Tier Configuration System**: Override → Overlay → Memory Cache → Persistent Cache → Autotune → Heuristics → Error
- **Automatic Platform Detection**: Seamlessly selects optimal implementation based on hardware
- **Priority-Based Registry**: Multi-backend support with intelligent fallback mechanisms
- **Device Fingerprinting**: Hardware-specific configuration caching for optimal performance

### State-of-the-Art Operations

- **15+ Attention Mechanisms**: Flash Attention v2, Ring Attention, Page Attention, Block Sparse, GLA, Lightning, and more
- **Memory Efficiency**: Custom VJP implementations with O(N) memory complexity for attention
- **Distributed Support**: Full shard_map integration for model and data parallelism
- **Mixed Precision**: Comprehensive dtype support with automatic gradient conversion

### Production-Ready Infrastructure

- **Type Safety**: Full jaxtyping annotations with runtime validation via beartype
- **Comprehensive Testing**: Cross-backend validation, performance benchmarks, integration tests
- **Atomic Persistence**: Thread-safe configuration storage with automatic optimization
- **Profiling Integration**: Built-in support for JAX profiling and performance monitoring

## Installation

### Basic Installation

```bash
pip install ejkernel
```

### Platform-Specific Installation

```bash
# GPU Support (CUDA/ROCm)
pip install ejkernel[gpu]

# TPU Support
pip install ejkernel[tpu]

# Development Installation
git clone https://github.com/erfanzar/ejkernel.git
cd ejkernel
pip install -e ".[dev]"
```

### Dependencies

- Python 3.11-3.13
- JAX >= 0.7.2
- Triton == 3.4.0 (for GPU)
- jaxtyping >= 0.3.2
- beartype >= 0.22.2

## Quick Start

### Simple API with Automatic Optimization

```python
import jax.numpy as jnp
from ejkernel.modules import flash_attention

# Basic usage - automatic configuration selection
output = flash_attention(
    query, key, value,
    causal=True,
    dropout_prob=0.1
)

# With advanced features
output = flash_attention(
    query, key, value,
    causal=True,
    sliding_window=128,        # Local attention window
    logits_soft_cap=30.0,     # Gemma-2 style soft capping
    attention_mask=mask,       # Custom attention pattern
)
```

### Custom Configuration

```python
from ejkernel.modules import FlashAttentionConfig
from ejkernel.ops.utils.datacarrier import FwdParams, BwdParams

# Create optimized configuration
config = FlashAttentionConfig(
    fwd_params=FwdParams(
        q_blocksize=256,
        kv_blocksize=256,
        num_warps=8,
        num_stages=2
    ),
    bwd_params=BwdParams(
        q_blocksize=128,
        kv_blocksize=128,
        num_warps=4
    ),
    platform="triton",  # Force specific backend
    backend="gpu"
)

output = flash_attention(query, key, value, cfg=config)
```

### Direct Kernel Registry Access

```python
from ejkernel import kernel_registry, Platform, Backend

# Get specific implementation
kernel = kernel_registry.get(
    algorithm="flash_attention",
    platform=Platform.TRITON,
    backend=Backend.GPU
)

# Direct execution
output = kernel(query, key, value, causal=True)
```

### Distributed Execution

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P
from ejkernel.modules import flash_attention

# Setup mesh for distributed execution
devices = jax.devices()
mesh = Mesh(devices, axis_names=("data", "model"))

# Run distributed attention
output = flash_attention(
    query, key, value,
    causal=True,
    mesh=mesh,
    in_specs=(P("data", None), P("data", None), P("data", None)),
    out_specs=P("data", None)
)
```

## Architecture Overview

### System Design

ejKernel employs a sophisticated layered architecture that separates concerns while maintaining high performance:

```md
┌─────────────────────────────────────────────────────┐
│              Public API (modules/)                   │
│         Simple functions with sensible defaults      │
├─────────────────────────────────────────────────────┤
│            Operations Layer (ops/)                   │
│    Configuration management, autotuning, caching     │
├─────────────────────────────────────────────────────┤
│          Kernel Registry (kernels/)                  │
│      Platform routing, signature validation          │
├─────────────────────────────────────────────────────┤
│      Backend Implementations (kernels/_*)            │
│         Triton, Pallas, XLA, CUDA kernels           │
└─────────────────────────────────────────────────────┘
```

### Project Structure

```md
ejkernel/
├── kernels/
│   ├── _triton/         # GPU kernels via Triton
│   ├── _pallas/         # TPU/GPU kernels via Pallas
│   │   ├── tpu/        # TPU-specific implementations
│   │   └── gpu/        # GPU Pallas implementations
│   ├── _xla/           # Universal XLA implementations
│   └── _cuda/          # Native CUDA kernels
├── modules/
│   └── operations/     # High-level API modules
├── ops/
│   ├── config/         # Configuration management
│   ├── core/           # Base kernel classes
│   ├── execution/      # Execution orchestration
│   └── utils/          # Fingerprinting, utilities
├── xla_utils/          # XLA-specific utilities
└── callib/             # Calibration utilities
```

### Core Components

#### Kernel Registry

The registry provides automatic platform-specific kernel selection:

```python
@kernel_registry.register("my_operation", Platform.TRITON, Backend.GPU, priority=100)
def my_operation_gpu(x, y):
    # GPU-optimized implementation
    pass

@kernel_registry.register("my_operation", Platform.XLA, Backend.ANY, priority=50)
def my_operation_fallback(x, y):
    # Universal fallback
    pass

# Automatic selection based on available hardware
impl = kernel_registry.get("my_operation")
```

#### Configuration Management

Multi-tier configuration system with intelligent fallback:

```python
class ConfigSelectorChain:
    """
    Selection hierarchy:
    1. Override - Explicit user configuration
    2. Overlay - Temporary context overrides
    3. Memory Cache - In-memory lookup
    4. Persistent Cache - Disk-based storage
    5. Autotune - Performance benchmarking
    6. Heuristics - Intelligent defaults
    7. Error - Clear failure message
    """
```

#### Custom VJP System

All performance-critical kernels implement memory-efficient gradients:

```python
@jax.custom_vjp
def kernel_with_custom_grad(inputs):
    return forward(inputs)

def kernel_fwd(inputs):
    output, residuals = forward_with_residuals(inputs)
    return output, residuals

def kernel_bwd(residuals, grad_output):
    return efficient_backward(residuals, grad_output)

kernel_with_custom_grad.defvjp(kernel_fwd, kernel_bwd)
```

## Supported Operations

### Attention Mechanisms

| Algorithm | Description | Memory | Key Features |
|-----------|-------------|--------|--------------|
| **Flash Attention v2** | Memory-efficient exact attention | O(N) | Causal masking, dropout, sliding windows, soft capping |
| **Ring Attention** | Distributed sequence parallelism | O(N/P) | Ultra-long sequences, communication overlap |
| **Page Attention** | KV-cache optimized inference | O(N) | Block-wise memory, continuous batching |
| **Block Sparse Attention** | Configurable sparse patterns | O(N√N) | Local+global, custom patterns |
| **GLA** | Gated Linear Attention | O(N) | Linear complexity, gated updates |
| **Lightning Attention** | Layer-dependent decay | O(N) | Exponential moving average |
| **MLA** | Multi-head Latent Attention | O(N) | Compressed KV representation |
| **Ragged Attention** | Variable-length sequences | O(N) | Efficient padding, batched inference |

### Other Operations

- **Recurrent Kernels**: Optimized RNN/LSTM/GRU operations
- **Mean Pooling**: Variable-length sequence aggregation
- **Grouped MatMul**: Efficient batched matrix operations
- **Native Sparse**: Block-sparse matrix computations

### Platform Support Matrix

| Operation | Triton (GPU) | Pallas (TPU) | XLA (Universal) | CUDA |
|-----------|-------------|--------------|-----------------|------|
| Flash Attention v2 | ✓ | ✓ | ✓ | Dev |
| Ring Attention | ✓ | ✓ | ✓ | Dev |
| Page Attention | ✓ | ✓ | ✓ | Dev |
| Block Sparse | ✓ | - | ✓ | Dev |
| GLA | ✓ | Dev | ✓ | - |
| Lightning | ✓ | - | ✓ | Dev |
| MLA | ✓ | Dev | - | - |
| Ragged Attention | ✓ | ✓ | ✓ | Dev |

✓ = Production ready | Dev = Under development | - = Not planned

## Advanced Usage

### Performance Optimization

```python
# Force autotuning for optimal configuration
import os
os.environ["EJKERNEL_AUTOTUNE_POLICY"] = "autotune"
os.environ["EJKERNEL_LOG_AUTOTUNE"] = "1"

# Enable profiling
os.environ["EJKERNEL_OPS_STAMP"] = "json"  # Detailed metadata
os.environ["EJKERNEL_OPS_RECORD"] = "1"    # Record invocations
```

### Custom Kernel Development

```python
from ejkernel.ops.core import Kernel
from ejkernel.modules.operations.configs import BaseOperationConfig

@dataclass
class MyConfig(BaseOperationConfig):
    param1: int = 128
    param2: float = 0.1

class MyKernel(Kernel[MyConfig, Array]):
    def __init__(self):
        super().__init__(op_id="my_kernel")

    def run(self, x, cfg: MyConfig):
        impl = kernel_registry.get("my_kernel", cfg.platform)
        return impl(x, param1=cfg.param1, param2=cfg.param2)

    def heuristic_cfg(self, inv):
        # Return default configuration
        return MyConfig(param1=256)

    def candidate_cfgs(self, inv):
        # Return autotuning candidates
        return [MyConfig(param1=p) for p in [64, 128, 256]]
```

### Integration with Models

```python
import flax.linen as nn

class TransformerBlock(nn.Module):
    num_heads: int = 8
    head_dim: int = 64

    @nn.compact
    def __call__(self, x, mask=None):
        # Project to Q, K, V
        q = nn.Dense(self.num_heads * self.head_dim)(x)
        k = nn.Dense(self.num_heads * self.head_dim)(x)
        v = nn.Dense(self.num_heads * self.head_dim)(x)

        # Reshape for attention
        shape = (x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        q, k, v = map(lambda t: t.reshape(shape), (q, k, v))

        # Apply ejKernel Flash Attention
        attn_output = flash_attention(
            q, k, v,
            causal=True,
            attention_mask=mask
        )

        # Project output
        return nn.Dense(x.shape[-1])(attn_output.reshape(x.shape))
```

## Performance

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/erfanzar/ejkernel.git
cd ejkernel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

The project uses:

- **black** for code formatting (line length: 121)
- **ruff** for linting
- **mypy/pyright** for type checking
- **pre-commit** for automated checks

### Adding New Kernels

1. **Implement the kernel** in appropriate backend directory:

```python
# ejkernel/kernels/_triton/my_kernel.py
@kernel_registry.register("my_kernel", Platform.TRITON, Backend.GPU)
def my_kernel_triton(x, config):
    # Implementation
    pass
```

2 **Create module wrapper**:

```python
# ejkernel/modules/operations/my_kernel.py
class MyKernel(Kernel[MyKernelConfig, Array]):
    # Module implementation
    pass
```

3 **Add tests**:

```python
# test/kernels/_triton/test_my_kernel.py
class TestMyKernel(unittest.TestCase):
    # Test implementation
    pass
```

4 **Update documentation**

## Testing

### Running Tests

```bash
# Run all tests
python test/run_tests.py

# Platform-specific tests
python test/run_tests.py --xla      # XLA implementations
python test/run_tests.py --triton   # Triton implementations
python test/run_tests.py --pallas   # Pallas implementations

# Cross-platform validation
python test/run_tests.py --comparison

# Specific test patterns
python test/run_tests.py -k "flash_attention"
python test/run_tests.py --verbose --failfast
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Comparison Tests**: Cross-backend consistency
- **Performance Tests**: Regression detection
- **Property Tests**: Invariant verification

### Continuous Integration

The project uses GitHub Actions for CI with tests across:

- Multiple Python versions (3.11, 3.12, 3.13)
- Multiple platforms (CPU, GPU, TPU)
- Multiple JAX versions

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Priority Areas

- TPU/Pallas implementations for existing algorithms
- CUDA native kernels for maximum performance
- New attention mechanisms from recent papers
- Performance optimizations and kernel fusion
- Documentation and examples

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## Documentation

Comprehensive documentation available at [ejkernel.readthedocs.io](https://ejkernel.readthedocs.io/en/latest/)

- **[API Reference](https://ejkernel.readthedocs.io/en/latest/api/)**: Complete API documentation
- **[Tutorials](https://ejkernel.readthedocs.io/en/latest/tutorials/)**: Step-by-step guides
- **[Architecture](https://ejkernel.readthedocs.io/en/latest/architecture/)**: Design documentation
- **[Benchmarks](https://ejkernel.readthedocs.io/en/latest/benchmarks/)**: Performance analysis

## Citation

If you use ejKernel in your research, please cite:

```bibtex
@software{ejkernel2024,
  author = {Erfan Zare Chavoshi},
  title = {ejKernel: High-Performance JAX Kernels for Deep Learning},
  year = {2024},
  url = {https://github.com/erfanzar/ejkernel},
  note = {Production-grade kernel library with multi-backend support}
}
```

## License

ejKernel is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

ejKernel builds upon excellent work from:

- [JAX](https://github.com/google/jax) - Composable transformations of Python+NumPy programs
- [Triton](https://github.com/openai/triton) - GPU kernel programming language
- [Pallas](https://github.com/google/jax/tree/main/jax/experimental/pallas) - JAX kernel language
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
- [EasyDeL](https://github.com/erfanzar/EasyDeL) - Parent framework for JAX deep learning

## Community

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/erfanzar/ejkernel/issues)
- **Discussions**: [Community forum](https://github.com/erfanzar/ejkernel/discussions)
- **Email**: <Erfanzare810@gmail.com>

## Roadmap

### Near Term (Q1 2025)

- Flash Attention 3 implementation
- Complete CUDA backend
- Quantized attention (INT8/INT4)
- Fused operations (LayerNorm+Attention)

### Medium Term (Q2-Q3 2025)

- Speculative decoding support
- Continuous batching
- Mamba SSM kernels

### Long Term (Q4 2025+)

- Multi-GPU kernel fusion
- Automatic kernel selection ML model
- Custom DSL for kernel development
- Hardware-agnostic IR

---

ejKernel - Production-grade kernels for JAX deep learning
