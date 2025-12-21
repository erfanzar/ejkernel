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


"""Backward pass implementation for SSM1 (Mamba1-style) selective state space.

Gradients for the SSM1 recurrence:
    Forward:
        dA = exp(A * dt)
        dBx = dt * B * x
        h_t = dA * h_{t-1} + dBx_t
        y_t = sum(h_t * C_t) + D * x_t

    Backward (computed in reverse order):
        dL/dh_t = dL/dy_t * C_t + dA_{t+1} * dL/dh_{t+1}
        dL/dx_t = dL/dy_t * D + sum(dL/dh_t * dt * B)
        dL/dA = sum_t(dL/dh_t * h_{t-1} * A * dt)
        dL/dB_t = sum_d(dL/dh_t * dt * x)
        dL/dC_t = sum_d(dL/dy_t * h_t)
        dL/dD = sum_t(dL/dy_t * x_t)
        dL/ddt_t = sum(dL/dh_t * (A * dA * h_{t-1} + B * x))
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _ssm1_bwd(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    all_hidden_states: Float[Array, "batch seq_len intermediate_size ssm_state_size"],
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"],
    do: Float[Array, "batch seq_len intermediate_size"],
    dfinal_state: Float[Array, "batch intermediate_size ssm_state_size"],
) -> tuple:
    """Backward pass for SSM1 recurrence.

    Args:
        hidden_states: Input tensor [batch, seq_len, intermediate_size]
        A: A matrix [intermediate_size, ssm_state_size]
        B: B parameter [batch, seq_len, ssm_state_size]
        C: C parameter [batch, seq_len, ssm_state_size]
        D: Skip connection [intermediate_size]
        dt: Time step [batch, seq_len, intermediate_size]
        all_hidden_states: Hidden states from forward [batch, seq_len, d, n]
        initial_state: Initial hidden state [batch, intermediate_size, ssm_state_size]
        do: Gradient of output [batch, seq_len, intermediate_size]
        dfinal_state: Gradient of final hidden state [batch, d, n]

    Returns:
        Tuple of (d_hidden_states, dA, dB, dC, dD, ddt, d_initial_state)
    """
    _batch_size, seq_len, _intermediate_size = hidden_states.shape

    def process_batch(
        x_b,
        A_val,
        B_b,
        C_b,
        D_val,
        dt_b,
        h_all_b,
        h0_b,
        do_b,
        dh_final,
    ):
        """Process backward for a single batch element."""

        # Get previous hidden states (shifted by 1)
        # h_prev[t] = h_all[t-1] for t > 0, h_prev[0] = h0
        h_prev = jnp.concatenate([h0_b[None, :, :], h_all_b[:-1]], axis=0)

        def backward_step(carry, inputs):
            """Single backward step through time."""
            dh_next = carry
            _t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t = inputs

            # Compute dA_t for this timestep
            dA_t = jnp.exp(A_val * dt_t[:, None])  # [d, n]

            # dL/dh_t from output gradient
            # y_t = sum(h_t * C_t, axis=-1) + D * x_t
            # dy/dh = C_t (broadcast over d dimension)
            dh_from_output = do_t[:, None] * C_t[None, :]  # [d, n]

            # Total gradient w.r.t. h_t
            dh_t = dh_next + dh_from_output

            # dL/dx_t
            # From output: dy/dx = D
            # From state update: dh/dx = dt * B (integrated over state dimension)
            dx_t = do_t * D_val + jnp.sum(dh_t * dt_t[:, None] * B_t[None, :], axis=-1)

            # dL/dB_t = sum over d of (dL/dh_t * dt * x)
            dB_t = jnp.sum(dh_t * dt_t[:, None] * x_t[:, None], axis=0)  # [n]

            # dL/dC_t = sum over d of (dL/dy_t * h_t)
            dC_t = jnp.sum(do_t[:, None] * h_t, axis=0)  # [n]

            # dL/ddt_t: gradient through discretization
            # dA = exp(A * dt), d(dA)/d(dt) = A * dA
            # dBx = dt * B * x, d(dBx)/d(dt) = B * x
            ddt_t = jnp.sum(dh_t * (A_val * dA_t * h_prev_t + B_t[None, :] * x_t[:, None]), axis=-1)

            # dL/dA_t contribution from this step
            # dA = exp(A * dt), d(dA)/dA = dt * dA
            dA_contrib = dh_t * dt_t[:, None] * dA_t * h_prev_t  # [d, n]

            # Propagate gradient to previous hidden state
            dh_prev = dh_t * dA_t

            outputs = (dx_t, dB_t, dC_t, ddt_t, dA_contrib)
            return dh_prev, outputs

        # Prepare scan inputs (reverse order)
        scan_inputs = (
            jnp.arange(seq_len)[::-1],
            x_b[::-1],
            B_b[::-1],
            C_b[::-1],
            dt_b[::-1],
            h_all_b[::-1],
            h_prev[::-1],
            do_b[::-1],
        )

        d_initial_state, outputs = jax.lax.scan(backward_step, dh_final, scan_inputs)

        dx_b, dB_b, dC_b, ddt_b, dA_contribs = outputs

        # Reverse outputs back to forward order
        dx_b = dx_b[::-1]
        dB_b = dB_b[::-1]
        dC_b = dC_b[::-1]
        ddt_b = ddt_b[::-1]
        dA_contribs = dA_contribs[::-1]

        # Sum dA contributions over sequence
        dA_b = jnp.sum(dA_contribs, axis=0)  # [d, n]

        # dD contribution from this batch
        dD_b = jnp.sum(do_b * x_b, axis=0)  # [d]

        return dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state

    # Process all batches - A and D are not batched, so use in_axes=None for them
    dx, dA_batch, dB, dC, dD_batch, ddt, d_initial_state = jax.vmap(
        process_batch,
        in_axes=(0, None, 0, 0, None, 0, 0, 0, 0, 0),
    )(
        hidden_states,
        A,  # not batched
        B,
        C,
        D,  # not batched
        dt,
        all_hidden_states,
        initial_state,
        do,
        dfinal_state,
    )

    # Sum dA and dD over batch dimension
    dA = jnp.sum(dA_batch, axis=0)
    dD = jnp.sum(dD_batch, axis=0)

    return dx, dA, dB, dC, dD, ddt, d_initial_state
