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


"""Test platform-specific kernel functionality."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from ejkernel.ops import ConfigCache, ConfigSelectorChain, Executor, Kernel, get_device_platform


@dataclass
class TestConfig:
    """Test configuration with different parameters per platform."""

    algorithm: str = "default"
    block_size: int = 128


class PlatformSpecificKernel(Kernel[TestConfig, jax.Array]):
    """Example kernel with platform-specific implementations."""

    def __init__(self):
        super().__init__("test_platform_kernel")

    # Generic fallback methods
    def run(self, x, y, cfg: TestConfig) -> jax.Array:
        """Generic implementation - simple addition."""
        return x + y + jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg(self, inv) -> TestConfig:
        """Generic heuristic configuration."""
        return TestConfig(algorithm="generic", block_size=128)

    def candidate_cfgs(self, inv):
        """Generic candidate configurations."""
        return [
            TestConfig(algorithm="generic", block_size=64),
            TestConfig(algorithm="generic", block_size=128),
        ]

    # GPU-specific methods
    def run_gpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """GPU-optimized implementation - uses multiplication."""
        return x * y * jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg_gpu(self, inv) -> TestConfig:
        """GPU-specific heuristic configuration."""
        return TestConfig(algorithm="gpu_optimized", block_size=256)

    def candidate_cfgs_gpu(self, inv):
        """GPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="gpu_fast", block_size=128),
            TestConfig(algorithm="gpu_optimized", block_size=256),
            TestConfig(algorithm="gpu_precise", block_size=512),
        ]

    # TPU-specific methods
    def run_tpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """TPU-optimized implementation - uses subtraction."""
        return x - y + jnp.array(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg_tpu(self, inv) -> TestConfig:
        """TPU-specific heuristic configuration."""
        return TestConfig(algorithm="tpu_optimized", block_size=1024)

    def candidate_cfgs_tpu(self, inv):
        """TPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="tpu_fast", block_size=512),
            TestConfig(algorithm="tpu_optimized", block_size=1024),
            TestConfig(algorithm="tpu_precise", block_size=2048),
        ]

    # CPU-specific methods
    def run_cpu(self, x, y, cfg: TestConfig) -> jax.Array:
        """CPU-optimized implementation - uses division."""
        return (x + y) / jnp.array(max(1, cfg.block_size), dtype=x.dtype)

    def heuristic_cfg_cpu(self, inv) -> TestConfig:
        """CPU-specific heuristic configuration."""
        return TestConfig(algorithm="cpu_optimized", block_size=64)

    def candidate_cfgs_cpu(self, inv):
        """CPU-specific candidate configurations."""
        return [
            TestConfig(algorithm="cpu_fast", block_size=32),
            TestConfig(algorithm="cpu_optimized", block_size=64),
        ]


class PlatformSpecificVJPKernel(Kernel[TestConfig, jax.Array]):
    """Example kernel with platform-specific VJP implementations."""

    def __init__(self):
        super().__init__("test_platform_vjp_kernel")

    # Generic implementations
    def run(self, x, y, cfg: TestConfig) -> jax.Array:
        return x @ y

    def heuristic_cfg(self, inv) -> TestConfig:
        return TestConfig(algorithm="generic")

    def fwd_with_residuals(self, x, y, cfg: TestConfig):
        result = x @ y
        return result, (x, y)

    def vjp(self, residuals, output, dy, *args, cfg: TestConfig, **kwargs):
        x, y = residuals
        dx = dy @ y.T
        dy_val = x.T @ dy
        return dx, dy_val

    # GPU-specific VJP
    def fwd_with_residuals_gpu(self, x, y, cfg: TestConfig):
        """GPU-optimized forward pass - potentially with different memory layout."""
        result = jnp.dot(x, y, precision=jax.lax.Precision.HIGH)
        return result, (x, y)

    def vjp_gpu(self, residuals, output, dy, *args, cfg: TestConfig, **kwargs):
        """GPU-optimized backward pass."""
        x, y = residuals
        dx = jnp.dot(dy, y.T, precision=jax.lax.Precision.HIGH)
        dy_val = jnp.dot(x.T, dy, precision=jax.lax.Precision.HIGH)
        return dx, dy_val


class TestPlatformSpecificKernels:
    """Test platform-specific kernel functionality."""

    def test_platform_detection(self):
        """Test that platform detection works correctly."""
        platform = get_device_platform()
        assert platform in ["gpu", "tpu", "cpu", "unknown"]

    def test_platform_specific_run(self):
        """Test that platform-specific run methods are selected."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificKernel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        result = executor(kernel, x, y)

        # Verify result shape
        assert result.shape == x.shape

        # The actual operation depends on the platform, but we can verify it ran
        assert jnp.all(jnp.isfinite(result))

    def test_platform_specific_candidate_cfgs(self):
        """Test that platform-specific candidate configs are used."""
        kernel = PlatformSpecificKernel()
        platform = get_device_platform()

        # Create dummy invocation
        from ejkernel.ops import Invocation

        inv = Invocation(
            op_id=kernel.op_id,
            args=(jnp.array([1.0]), jnp.array([2.0])),
            kwargs={},
        )

        # Get platform-specific method
        from ejkernel.ops.core import _get_platform_method

        candidate_method = _get_platform_method(kernel, "candidate_cfgs", platform)

        if candidate_method:
            # Platform-specific method exists
            configs = list(candidate_method(inv))
            # Check that we got platform-specific configs
            if platform == "gpu":
                assert any("gpu" in cfg.algorithm for cfg in configs)
            elif platform == "tpu":
                assert any("tpu" in cfg.algorithm for cfg in configs)
            elif platform == "cpu":
                assert any("cpu" in cfg.algorithm for cfg in configs)
        else:
            # Fall back to generic
            configs = list(kernel.candidate_cfgs(inv))
            assert any("generic" in cfg.algorithm for cfg in configs)

    def test_platform_specific_heuristic(self):
        """Test that platform-specific heuristic configs are used."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificKernel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Get the chosen config
        cfg = executor.choose_config(kernel, x, y)

        # Verify platform-specific config was chosen
        platform = get_device_platform()
        if platform == "gpu":
            assert "gpu" in cfg.algorithm
        elif platform == "tpu":
            assert "tpu" in cfg.algorithm
        elif platform == "cpu":
            assert "cpu" in cfg.algorithm
        else:
            # Unknown platform falls back to generic
            assert cfg.algorithm == "generic"

    def test_platform_specific_vjp(self):
        """Test that platform-specific VJP methods work correctly."""
        cache = ConfigCache()
        selector = ConfigSelectorChain(cache)
        executor = Executor(selector)

        kernel = PlatformSpecificVJPKernel()
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        # Test forward pass
        result = executor(kernel, x, y)
        assert result.shape == (2, 2)

        # Test gradient computation
        def f(x_val, y_val):
            return jnp.sum(executor(kernel, x_val, y_val))

        grad_fn = jax.grad(f, argnums=(0, 1))
        dx, dy = grad_fn(x, y)

        # Verify gradients have correct shape
        assert dx.shape == x.shape
        assert dy.shape == y.shape

        # Verify gradients are finite
        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dy))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
