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


"""SSM1 (Mamba1-style) Selective State Space operation module.

This module provides the StateSpaceV1 operation, implementing the original Mamba
selective state space model architecture used by Mamba and FalconMamba.

Key characteristics of SSM1:
- 2D A matrix: [intermediate_size, ssm_state_size]
- SSM state shape: [batch, intermediate_size, ssm_state_size]
- Separate dt_proj projection for time step
- Output gating: y * activation(gate)

The algorithm:
    Discretization:
        dA = exp(A * dt)
        dB = dt * B

    Recurrence:
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t

Features:
    - O(N) complexity through sequential processing
    - Stateful computation with hidden state propagation
    - Optional gating with configurable activation
    - Conv state passthrough for caching

References:
    - Mamba: https://arxiv.org/abs/2312.00752
    - FalconMamba: https://huggingface.co/tiiuae/falcon-mamba-7b
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable
from typing import Literal

from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import StateSpaceV1Config


class StateSpaceV1(Kernel[StateSpaceV1Config, Array]):
    """SSM1 (Mamba1-style) Selective State Space operation.

    Implements the original Mamba architecture with O(N) complexity.
    Processes tokens sequentially, maintaining a hidden state that
    accumulates information through discretized state transitions.

    Features:
        - 2D A matrix [intermediate_size, ssm_state_size]
        - Hidden state propagation across timesteps
        - Optional gating with activation function
        - Conv state passthrough for caching
        - Multiple platform support (XLA primary)

    The state update mechanism:
        dA = exp(A * dt)
        dB = dt * B
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t
    """

    def __init__(self):
        """Initialize StateSpaceV1 module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="state_space_v1")

    def get_impl(self, cfg: StateSpaceV1Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for SSM1

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("state_space_v1", cfg.platform)
        return kernel_registry.get("state_space_v1", platform=platform, backend=cfg.backend)

    def run(
        self,
        hidden_states: Float[Array, "batch seq_len intermediate_size"],
        A: Float[Array, "intermediate_size ssm_state_size"],
        B: Float[Array, "batch seq_len ssm_state_size"],
        C: Float[Array, "batch seq_len ssm_state_size"],
        D: Float[Array, "intermediate_size"],
        dt: Float[Array, "batch seq_len intermediate_size"],
        gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
        initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
        conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
        act_fn: Callable[[Array], Array] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: StateSpaceV1Config,
    ) -> tuple[
        Float[Array, "batch seq_len intermediate_size"],
        Float[Array, "batch intermediate_size ssm_state_size"],
        Float[Array, "batch intermediate_size d_conv"] | None,
    ]:
        """Execute SSM1 selective state space operation.

        Args:
            hidden_states: Input tensor after convolution and activation
                Shape: [batch, seq_len, intermediate_size]
            A: A matrix in real form (typically negative for stability)
                Shape: [intermediate_size, ssm_state_size]
            B: B parameter from input projection
                Shape: [batch, seq_len, ssm_state_size]
            C: C parameter from input projection
                Shape: [batch, seq_len, ssm_state_size]
            D: Skip connection parameter
                Shape: [intermediate_size]
            dt: Time step after softplus activation
                Shape: [batch, seq_len, intermediate_size]
            gate: Optional gating tensor for output modulation
                Shape: [batch, seq_len, intermediate_size]
            initial_state: Optional initial SSM state for continuation
                Shape: [batch, intermediate_size, ssm_state_size]
            conv_state: Optional convolution state for caching (passed through)
                Shape: [batch, intermediate_size, d_conv]
            act_fn: Optional activation function for gating (e.g., jax.nn.silu)
            platform: Optional platform override
            cfg: Kernel configuration object

        Returns:
            Tuple of:
                - output: SSM output [batch, seq_len, intermediate_size]
                - ssm_state: Final hidden state [batch, intermediate_size, ssm_state_size]
                - conv_state: Passed through conv_state (for caching)
        """
        if platform is not None:
            cfg = StateSpaceV1Config(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        return impl(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=D,
            dt=dt,
            gate=gate,
            initial_state=initial_state,
            conv_state=conv_state,
            act_fn=act_fn,
        )

    def heuristic_cfg(self, inv: Invocation[StateSpaceV1Config, Array]) -> StateSpaceV1Config:
        """Provide default configuration.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration for SSM1 operation
        """
        return StateSpaceV1Config(
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[StateSpaceV1Config, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            SSM1 uses XLA implementation primarily, so candidates are minimal.
        """
        return [
            StateSpaceV1Config(platform="auto", backend="any"),
            StateSpaceV1Config(platform="xla", backend="any"),
        ]


_state_space_v1_executor: Executor[StateSpaceV1Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("state_space_v1"),
    )
)


def state_space_v1(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    /,
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
    *,
    act_fn: Callable[[Array], Array] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: StateSpaceV1Config | None = None,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
    Float[Array, "batch intermediate_size d_conv"] | None,
]:
    """Execute SSM1 (Mamba1-style) selective state space with automatic optimization.

    SSM1 processes sequences with stateful computation, maintaining hidden states
    across timesteps for O(N) complexity selective state space modeling.

    Args:
        hidden_states: Input tensor after convolution and activation
            Shape: [batch, seq_len, intermediate_size]
        A: A matrix in real form (typically negative for stability)
            Shape: [intermediate_size, ssm_state_size]
        B: B parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        C: C parameter from input projection
            Shape: [batch, seq_len, ssm_state_size]
        D: Skip connection parameter
            Shape: [intermediate_size]
        dt: Time step after softplus activation
            Shape: [batch, seq_len, intermediate_size]
        gate: Optional gating tensor for output modulation
            Shape: [batch, seq_len, intermediate_size]
        initial_state: Optional initial SSM state for continuation
            Shape: [batch, intermediate_size, ssm_state_size]
        conv_state: Optional convolution state for caching (passed through)
            Shape: [batch, intermediate_size, d_conv]
        act_fn: Optional activation function for gating (e.g., jax.nn.silu).
            If gate is provided but act_fn is None, defaults to jax.nn.silu.
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional kernel configuration

    Returns:
        Tuple of:
            - output: SSM output [batch, seq_len, intermediate_size]
            - ssm_state: Final hidden state [batch, intermediate_size, ssm_state_size]
            - conv_state: Passed through conv_state (for caching)

    Example:
        >>> # Basic usage
        >>> output, ssm_state, _ = state_space_v1(hidden_states, A, B, C, D, dt)
        >>>
        >>> # With gating
        >>> output, ssm_state, _ = state_space_v1(
        ...     hidden_states, A, B, C, D, dt,
        ...     gate=gate, act_fn=jax.nn.silu,
        ... )
        >>>
        >>> # Inference with cached state
        >>> output, new_state, conv_state = state_space_v1(
        ...     hidden_states[:, :1, :],
        ...     A, B[:, :1, :], C[:, :1, :], D, dt[:, :1, :],
        ...     initial_state=ssm_state, conv_state=conv_state,
        ... )
    """
    return _state_space_v1_executor(
        StateSpaceV1(),
        hidden_states=hidden_states,
        A=A,
        B=B,
        C=C,
        D=D,
        dt=dt,
        gate=gate,
        initial_state=initial_state,
        conv_state=conv_state,
        act_fn=act_fn,
        platform=platform,
        _cfg=cfg,
    )
