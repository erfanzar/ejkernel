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


"""Backward pass implementation for SSM2 (Mamba2-style) selective state space.

Gradients for the SSM2 recurrence:
    Forward:
        dA = exp(A * dt)
        dBx = dt * B * x (outer product)
        h_t = dA * h_{t-1} + dBx
        y_t = einsum(h_t, C_t) + x_t * D

    Backward (computed in reverse order):
        dL/dh_t = einsum(dL/dy_t, C_t) + dA_{t+1} * dL/dh_{t+1}
        dL/dx_t = dL/dy_t * D + einsum(dL/dh_t, dt * B)
        dL/dA = sum_t(dL/dh_t * h_{t-1} * dt * dA)
        dL/dB_t = einsum(dL/dh_t, x_t * dt)
        dL/dC_t = einsum(dL/dy_t, h_t)
        dL/dD = sum_t(dL/dy_t * x_t)
        dL/ddt_t = sum(dL/dh_t * (A * dA * h_{t-1} + B * x))
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _ssm2_bwd(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len num_heads ssm_state_size"],
    C: Float[Array, "batch seq_len num_heads ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    all_hidden_states: Float[Array, "batch seq_len num_heads head_dim ssm_state_size"],
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
    do: Float[Array, "batch seq_len num_heads head_dim"],
    dfinal_state: Float[Array, "batch num_heads head_dim ssm_state_size"],
) -> tuple:
    """Backward pass for SSM2 recurrence.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        A: A vector [num_heads]
        B: B parameter [batch, seq_len, num_heads, ssm_state_size]
        C: C parameter [batch, seq_len, num_heads, ssm_state_size]
        D: Skip connection [num_heads]
        dt: Time step [batch, seq_len, num_heads]
        all_hidden_states: Hidden states from forward [batch, seq_len, h, d, n]
        initial_state: Initial hidden state [batch, num_heads, head_dim, ssm_state_size]
        do: Gradient of output [batch, seq_len, num_heads, head_dim]
        dfinal_state: Gradient of final hidden state [batch, h, d, n]

    Returns:
        Tuple of (dx, dA, dB, dC, dD, ddt, d_initial_state)
    """
    _batch_size, seq_len, _num_heads, _head_dim = x.shape

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
        h_prev = jnp.concatenate([h0_b[None, :, :, :], h_all_b[:-1]], axis=0)

        def backward_step(carry, inputs):
            """Single backward step through time."""
            dh_next = carry
            _t_idx, x_t, B_t, C_t, dt_t, h_t, h_prev_t, do_t = inputs
            # x_t: [num_heads, head_dim]
            # B_t: [num_heads, ssm_state_size]
            # C_t: [num_heads, ssm_state_size]
            # dt_t: [num_heads]
            # h_t: [num_heads, head_dim, ssm_state_size]
            # do_t: [num_heads, head_dim]

            # Compute dA_t for this timestep
            dA_t = jnp.exp(dt_t[:, None, None] * A_val[:, None, None])  # [h, 1, 1]

            # dL/dh_t from output gradient
            # y_t = einsum("hdn,hn->hd", h_t, C_t)
            # dy/dh_t[h,d,n] = do_t[h,d] * C_t[h,n]
            dh_from_output = do_t[:, :, None] * C_t[:, None, :]  # [h, d, n]

            # Total gradient w.r.t. h_t
            dh_t = dh_next + dh_from_output

            # dL/dx_t
            # From output: dy/dx = D
            # From state update: dh/dx = dt * B
            # dh_t: [h, d, n], dt_t: [h], B_t: [h, n]
            dx_t = do_t * D_val[:, None] + jnp.einsum("hdn,h,hn->hd", dh_t, dt_t, B_t)

            # dL/dB_t
            # dBx = dt * B * x (outer product), dBx[h,d,n] = dt[h] * B[h,n] * x[h,d]
            # dL/dB[h,n] = sum_d(dL/dh[h,d,n] * dt[h] * x[h,d])
            dB_t = jnp.einsum("hdn,h,hd->hn", dh_t, dt_t, x_t)

            # dL/dC_t
            # y_t = einsum("hdn,hn->hd", h_t, C_t)
            # dL/dC[h,n] = sum_d(dL/dy[h,d] * h_t[h,d,n])
            dC_t = jnp.einsum("hd,hdn->hn", do_t, h_t)

            # dL/ddt_t: gradient through discretization
            # dA = exp(A * dt), d(dA)/d(dt) = A * dA
            # dBx = dt * B * x, d(dBx)/d(dt) = B * x
            # dL/ddt[h] = sum_{d,n}(dL/dh[h,d,n] * (A[h] * dA[h] * h_prev[h,d,n] + B[h,n] * x[h,d]))
            ddt_t = jnp.sum(
                dh_t * (A_val[:, None, None] * dA_t * h_prev_t + B_t[:, None, :] * x_t[:, :, None]),
                axis=(-2, -1),
            )

            # dL/dA_t contribution from this step
            # dA = exp(A * dt), d(dA)/dA = dt * dA
            # dL/dA[h] = sum_{d,n}(dL/dh[h,d,n] * dt[h] * dA[h] * h_prev[h,d,n])
            dA_contrib = jnp.sum(dh_t * dt_t[:, None, None] * dA_t * h_prev_t, axis=(-2, -1))

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
        dA_b = jnp.sum(dA_contribs, axis=0)  # [h]

        # dD contribution from this batch
        dD_b = jnp.sum(jnp.sum(do_b * x_b, axis=0), axis=-1)  # [h]

        return dx_b, dA_b, dB_b, dC_b, dD_b, ddt_b, d_initial_state

    # Process all batches - A and D are not batched, so use in_axes=None for them
    dx, dA_batch, dB, dC, dD_batch, ddt, d_initial_state = jax.vmap(
        process_batch,
        in_axes=(0, None, 0, 0, None, 0, 0, 0, 0, 0),
    )(
        x,
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
