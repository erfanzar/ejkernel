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


"""Forward pass implementation for SSM2 (Mamba2-style) selective state space.

SSM2 algorithm:
    Discretization:
        dA = exp(A * dt)  where A is per-head
        dB = dt * B

    Recurrence (per head):
        dBx = dt * B * x (outer product form)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

Key characteristics:
    - 1D A vector: [num_heads] (per-head scalar)
    - SSM state shape: [batch, num_heads, head_dim, ssm_state_size]
    - B, C with n_groups grouping
"""

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


def _ssm2_step(
    carry: tuple[Float[Array, "num_heads head_dim ssm_state_size"]],
    inputs: tuple,
    A: Float[Array, "num_heads"],
    D: Float[Array, "num_heads"],
) -> tuple[
    tuple[Float[Array, "num_heads head_dim ssm_state_size"]],
    Float[Array, "num_heads head_dim"],
]:
    """Single step of SSM2 recurrence.

    Updates hidden state: h_t = dA * h_{t-1} + dBx
    Computes output: y_t = einsum(h_t, C_t) + x_t * D

    Args:
        carry: Hidden state (h,) where h is [num_heads, head_dim, ssm_state_size]
        inputs: Tuple of (x_t, B_t, C_t, dt_t) for current timestep
        A: A vector (per-head, negative) [num_heads]
        D: Skip connection [num_heads]

    Returns:
        Updated carry and output for this timestep
    """
    (h,) = carry
    x_t, B_t, C_t, dt_t = inputs
    # x_t: [num_heads, head_dim]
    # B_t: [num_heads, ssm_state_size]
    # C_t: [num_heads, ssm_state_size]
    # dt_t: [num_heads]

    # dA = exp(dt * A) where A is [num_heads]
    # dt_t: [num_heads] -> [num_heads, 1, 1]
    dA = jnp.exp(dt_t[:, None, None] * A[:, None, None])

    # dBx = dt * B * x (outer product form)
    # [num_heads, head_dim, ssm_state_size]
    dBx = (dt_t[:, None, None] * B_t[:, None, :]) * x_t[:, :, None]

    # State update
    new_h = h * dA + dBx

    # Output: einsum("hdn,hn->hd", new_h, C_t)
    y_t = jnp.einsum("hdn,hn->hd", new_h, C_t)

    # Skip connection
    y_t = y_t + x_t * D[:, None]

    return (new_h,), y_t


def _ssm2_fwd(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len num_heads ssm_state_size"],
    C: Float[Array, "batch seq_len num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"],
    Float[Array, "batch seq_len num_heads head_dim ssm_state_size"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
]:
    """Forward pass for SSM2 recurrence.

    Processes sequences with O(N) complexity by maintaining a hidden state.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        A: A vector (real form, negative) [num_heads]
        B: B parameter [batch, seq_len, num_heads, ssm_state_size]
        C: C parameter [batch, seq_len, num_heads, ssm_state_size]
        D: Skip connection [num_heads]
        dt: Time step (after softplus) [batch, seq_len, num_heads]
        initial_state: Initial hidden state [batch, num_heads, head_dim, ssm_state_size]

    Returns:
        Tuple of (output, all_hidden_states, final_state)
    """
    batch_size, _seq_len, num_heads, head_dim = x.shape
    ssm_state_size = B.shape[-1]

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch_size, num_heads, head_dim, ssm_state_size),
            dtype=jnp.float32,
        )

    def process_batch(x_b, B_b, C_b, dt_b, h0):
        """Process a single batch element."""

        def scan_fn(carry, inputs):
            x_t, B_t, C_t, dt_t = inputs
            (h,) = carry

            # dA = exp(dt * A)
            dA = jnp.exp(dt_t[:, None, None] * A[:, None, None])

            # dBx = dt * B * x (outer product)
            dBx = (dt_t[:, None, None] * B_t[:, None, :]) * x_t[:, :, None]

            # State update
            new_h = h * dA + dBx

            # Output
            y_t = jnp.einsum("hdn,hn->hd", new_h, C_t) + x_t * D[:, None]

            return (new_h,), (new_h, y_t)

        scan_inputs = (x_b, B_b, C_b, dt_b)
        (h_final,), (hidden_states_all, outputs) = lax.scan(scan_fn, (h0,), scan_inputs)

        return outputs, hidden_states_all, h_final

    outputs, all_hidden_states, final_states = jax.vmap(process_batch)(
        x,  # [batch, seq_len, num_heads, head_dim]
        B,  # [batch, seq_len, num_heads, ssm_state_size]
        C,  # [batch, seq_len, num_heads, ssm_state_size]
        dt,  # [batch, seq_len, num_heads]
        initial_state,  # [batch, num_heads, head_dim, ssm_state_size]
    )

    return outputs, all_hidden_states, final_states


def _ssm2_single_step_fwd(
    x: Float[Array, "batch num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch num_heads ssm_state_size"],
    C: Float[Array, "batch num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch num_heads"],
    ssm_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
) -> tuple[
    Float[Array, "batch num_heads head_dim"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
]:
    """Single step SSM2 update for inference.

    Args:
        x: Current input [batch, num_heads, head_dim]
        A: A vector (real form) [num_heads]
        B: B parameter [batch, num_heads, ssm_state_size]
        C: C parameter [batch, num_heads, ssm_state_size]
        D: Skip connection [num_heads]
        dt: Time step [batch, num_heads]
        ssm_state: Previous state [batch, num_heads, head_dim, ssm_state_size]

    Returns:
        Tuple of (output, new_state)
    """
    # dA = exp(dt * A)
    # dt: [batch, num_heads] -> [batch, num_heads, 1, 1]
    dA = jnp.exp(dt[:, :, None, None] * A[None, :, None, None])

    # dBx = dt * B * x (outer product)
    # [batch, num_heads, head_dim, ssm_state_size]
    dBx = (dt[:, :, None, None] * B[:, :, None, :]) * x[:, :, :, None]

    # State update
    new_state = ssm_state * dA + dBx

    # Output: einsum("bhdn,bhn->bhd", new_state, C)
    y = jnp.einsum("bhdn,bhn->bhd", new_state, C)

    # Skip connection
    y = y + x * D[None, :, None]

    return y, new_state
