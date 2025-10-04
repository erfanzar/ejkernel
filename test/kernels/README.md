# ejkernels Test Suite

Comprehensive test suite for ejkernels XLA and Triton implementations.

## Structure

```md
test/kernels/
├── _xla/                   # XLA implementation tests
│   ├── test_recurrent.py       # Recurrent attention, GLA, Lightning
│   ├── test_sparse_attention.py # Native sparse attention
│   ├── test_mean_pooling.py    # Mean pooling (fixed & varlen)
│   └── test_page_attention.py  # Page attention (inference)
├── _triton/                # Triton implementation tests
│   └── test_flash_attention.py # Flash attention
├── comparison/             # XLA vs Triton comparison tests
│   └── test_xla_vs_triton.py   # Numerical accuracy comparisons
└── README.md
```

## Running Tests

### All Tests

```bash
python test/run_tests.py
```

### XLA Tests Only

```bash
python test/run_tests.py --xla
```

### Triton Tests Only

```bash
python test/run_tests.py --triton
```

### Comparison Tests Only

```bash
python test/run_tests.py --comparison
```

### Specific Test

```bash
python test/run_tests.py -k test_forward_shape
```

### Verbose Output

```bash
python test/run_tests.py --verbose
```

### Stop on First Failure

```bash
python test/run_tests.py --failfast
```

## Test Categories

### XLA Kernel Tests

#### Recurrent Attention (`test_recurrent.py`)

- ✅ Forward pass shape correctness
- ✅ Gradient computation and shapes
- ✅ Orthogonal vector tests
- ✅ Initial state handling
- ✅ Reverse mode processing
- ✅ GLA (Gated Linear Attention)
- ✅ Lightning Attention (layer-dependent decay)

#### Native Sparse Attention (`test_sparse_attention.py`)

- ✅ Forward pass with block sparsity
- ✅ Gradient computation
- ✅ Local attention patterns
- ✅ Custom attention scales
- ✅ Variable block counts

#### Mean Pooling (`test_mean_pooling.py`)

- ✅ Fixed-length pooling
- ✅ Variable-length pooling with cu_seqlens
- ✅ Gradient distribution
- ✅ Numerical correctness

#### Page Attention (`test_page_attention.py`)

- ✅ Basic paged attention
- ✅ GQA (Grouped Query Attention) support
- ✅ Different context lengths
- ✅ Block table indexing

### Triton Kernel Tests

#### Flash Attention (`test_flash_attention.py`)

- ✅ Forward and backward passes
- ✅ Causal masking
- ✅ Custom scales

### Comparison Tests (`test_xla_vs_triton.py`)

#### Numerical Accuracy

- ✅ Recurrent: XLA vs Triton outputs
- ✅ GLA: XLA vs Triton outputs
- ✅ Lightning Attention: XLA vs Triton outputs
- ✅ Mean Pooling: XLA vs Triton outputs
- ✅ Random input validation
- ✅ Relative error checking (< 1e-3 for attention, < 1e-5 for pooling)

## Test Coverage

### XLA Implementations

- ✅ `recurrent` - Custom backward with stored hidden states
- ✅ `recurrent_gla` - Inherits from recurrent
- ✅ `lightning_attn` - Inherits from recurrent
- ✅ `mean_pooling` - Custom VJP
- ✅ `apply_native_sparse_attention` - Custom VJP
- ✅ `page_attention` - Inference only (no gradients tested)
- ✅ `ragged_page_attention` - Inference only

### Triton Implementations

- ✅ `flash_attention` - Full forward/backward
- ✅ `recurrent` - Compared against XLA
- ✅ `gla` - Compared against XLA
- ✅ `lightning_attn` - Compared against XLA
- ✅ `mean_pooling` - Compared against XLA

## Test Requirements

All tests use:

- `pytest` for test framework
- `jax` for automatic differentiation testing
- `jax.numpy` for array operations

## Adding New Tests

1. Create test file in appropriate directory (`_xla/`, `_triton/`, or `comparison/`)
2. Follow naming convention: `test_<kernel_name>.py`
3. Use pytest class structure for organization
4. Include tests for:
   - Forward pass shapes
   - Gradient shapes (for trainable kernels)
   - Numerical correctness
   - Edge cases
5. Add comparison test if both XLA and Triton implementations exist

## Example Test

```python
class TestMyKernel:
    def test_forward_shape(self):
        """Test output shapes are correct."""
        # Setup inputs
        x = jnp.ones((batch, seq_len, hidden_dim))

        # Run kernel
        output = my_kernel(x)

        # Check shape
        assert output.shape == expected_shape

    def test_gradients(self):
        """Test gradient computation."""
        def loss_fn(x):
            return jnp.sum(my_kernel(x))

        dx = jax.grad(loss_fn)(x)
        assert dx.shape == x.shape
```

## CI/CD Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run XLA tests
  run: python test/run_tests.py --xla

- name: Run comparison tests
  run: python test/run_tests.py --comparison
```
