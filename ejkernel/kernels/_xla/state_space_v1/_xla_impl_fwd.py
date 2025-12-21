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


"""Forward pass implementation for SSM1 (Mamba1-style) selective state space.

SSM1 algorithm:
    Discretization:
        dA = exp(A * dt)
        dB = dt * B

    Recurrence:
        h_t = dA * h_{t-1} + dB * x_t
        y_t = h_t @ C_t + D * x_t

Key characteristics:
    - 2D A matrix: [intermediate_size, ssm_state_size]
    - SSM state shape: [batch, intermediate_size, ssm_state_size]
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


def _ssm1_step(
    carry: tuple[Float[Array, "intermediate_size ssm_state_size"]],
    inputs: tuple,
    D: Float[Array, "intermediate_size"],
) -> tuple[tuple[Float[Array, "intermediate_size ssm_state_size"]], Float[Array, "intermediate_size"]]:
    """Single step of SSM1 recurrence.

    Updates hidden state: h_t = dA_t * h_{t-1} + dB_t * x_t
    Computes output: y_t = h_t @ C_t + D * x_t

    Args:
        carry: Hidden state (h,) where h is [intermediate_size, ssm_state_size]
        inputs: Tuple of (x_t, dA_t, dB_t, C_t) for current timestep
        D: Skip connection [intermediate_size]

    Returns:
        Updated carry and output for this timestep
    """
    (h,) = carry
    x_t, dA_t, dB_t, C_t = inputs

    # State update: h_t = dA_t * h_{t-1} + dB_t * x_t
    # dA_t: [intermediate_size, ssm_state_size]
    # dB_t: [intermediate_size, ssm_state_size] (already includes x_t multiplication)
    new_h = dA_t * h + dB_t

    # Output: y_t = h_t @ C_t + D * x_t
    # new_h: [intermediate_size, ssm_state_size]
    # C_t: [ssm_state_size]
    y_t = jnp.sum(new_h * C_t[None, :], axis=-1) + D * x_t

    return (new_h,), y_t


def _ssm1_fwd(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch seq_len intermediate_size ssm_state_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
]:
    """Forward pass for SSM1 recurrence.

    Processes sequences with O(N) complexity by maintaining a hidden state.

    Args:
        hidden_states: Input tensor [batch, seq_len, intermediate_size]
        A: A matrix (already in real form, negative) [intermediate_size, ssm_state_size]
        B: B parameter [batch, seq_len, ssm_state_size]
        C: C parameter [batch, seq_len, ssm_state_size]
        D: Skip connection [intermediate_size]
        dt: Time step (already after softplus) [batch, seq_len, intermediate_size]
        initial_state: Initial hidden state [batch, intermediate_size, ssm_state_size]

    Returns:
        Tuple of (output, all_hidden_states, final_state)
    """
    batch_size, _seq_len, intermediate_size = hidden_states.shape
    ssm_state_size = B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, intermediate_size, ssm_state_size),
            dtype=jnp.float32,
        )

    # Discretization
    # dA = exp(A * dt) where A is [d, n] and dt is [batch, seq_len, d]
    # Result: [batch, seq_len, d, n]
    discrete_A = jnp.exp(A[None, None, :, :] * dt[:, :, :, None])

    # dB * x = (dt * B) * x
    # dt: [batch, seq_len, d], B: [batch, seq_len, n]
    # Result: [batch, seq_len, d, n]
    discrete_Bx = dt[:, :, :, None] * B[:, None, :, :].swapaxes(1, 2) * hidden_states[:, :, :, None]

    def process_batch(x_b, dA_b, dBx_b, C_b, h0):
        """Process a single batch element."""

        def scan_fn(carry, inputs):
            x_t, dA_t, dBx_t, C_t = inputs
            (h,) = carry

            # State update
            new_h = dA_t * h + dBx_t

            # Output
            y_t = jnp.sum(new_h * C_t[None, :], axis=-1) + D * x_t

            return (new_h,), (new_h, y_t)

        scan_inputs = (x_b, dA_b, dBx_b, C_b)
        (h_final,), (hidden_states_all, outputs) = lax.scan(scan_fn, (h0,), scan_inputs)

        return outputs, hidden_states_all, h_final

    outputs, all_hidden_states, final_states = jax.vmap(process_batch)(
        hidden_states,  # [batch, seq_len, d]
        discrete_A,  # [batch, seq_len, d, n]
        discrete_Bx,  # [batch, seq_len, d, n]
        C,  # [batch, seq_len, n]
        initial_state,  # [batch, d, n]
    )

    return outputs, all_hidden_states, final_states


def _ssm1_single_step_fwd(
    hidden_state: Float[Array, "batch intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch ssm_state_size"],
    C: Float[Array, "batch ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch intermediate_size"],
    ssm_state: Float[Array, "batch intermediate_size ssm_state_size"],
) -> tuple[
    Float[Array, "batch intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
]:
    """Single step SSM1 update for inference.

    Args:
        hidden_state: Current input [batch, intermediate_size]
        A: A matrix (real form) [intermediate_size, ssm_state_size]
        B: B parameter [batch, ssm_state_size]
        C: C parameter [batch, ssm_state_size]
        D: Skip connection [intermediate_size]
        dt: Time step [batch, intermediate_size]
        ssm_state: Previous state [batch, intermediate_size, ssm_state_size]

    Returns:
        Tuple of (output, new_state)
    """
    # Discretization
    # dA = exp(A * dt)
    discrete_A = jnp.exp(A[None, :, :] * dt[:, :, None])  # [batch, d, n]

    # dB * x = dt * B * x
    discrete_Bx = dt[:, :, None] * B[:, None, :] * hidden_state[:, :, None]  # [batch, d, n]

    # State update
    new_state = discrete_A * ssm_state + discrete_Bx

    # Output: y = h @ C + D * x
    y = jnp.sum(new_state * C[:, None, :], axis=-1) + D[None, :] * hidden_state

    return y, new_state
