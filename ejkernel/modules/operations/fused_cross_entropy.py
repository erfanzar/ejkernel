# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Fused cross-entropy operation with automatic platform & sharding dispatch.

Computes per-row cross-entropy ``-log p(target | logits)`` (or its
soft-target generalisation) together with the analytic gradient
``softmax - target_dist``, fused into a single sweep of the vocabulary
so the ``[..., V]`` log-softmax / softmax tensor is never materialised
in HBM. Supports ``label_smoothing``, z-loss regularisation, and dense
soft targets.

The operation routes to whichever registered backend the platform
dispatcher selects (e.g. ``tilelang`` on NVIDIA GPUs, ``xla`` as the
portable fallback on TPU/CPU/AMD); additional backends can be
registered without touching this file.

When ``mesh`` / ``in_specs`` / ``out_specs`` are provided the call is
wrapped in :func:`jax.shard_map` automatically: vocab parallelism is
auto-detected from the last entry of the logits' partition spec, and
the loss/count ``psum`` reduction is auto-inserted over the leading
sharded axes.
"""

from __future__ import annotations

import os
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

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
from .configs import FusedCrossEntropyConfig

PlatformName = Literal["tilelang", "xla", "triton", "pallas", "cuda", "cute", "auto"] | str


class CrossEntropyOutput(NamedTuple):
    """Per-call cross-entropy metrics returned by :func:`fused_cross_entropy`.

    Attributes:
        loss: Differentiable scalar (``reduction in {"mean", "sum"}``) or
            ``logits.shape[:-1]`` array (``reduction == "none"``). This is
            the quantity to backprop through; the other fields are
            ``stop_gradient`` metrics.
        z_loss: ``z_loss · mean(lse²)`` (or ``sum`` / per-row, matching
            ``reduction``). Zero when the ``z_loss`` coefficient is 0.
        weight_sum: Sum of the per-token weights actually used by the
            kernel (the denominator of ``"mean"`` reduction).
        accuracy: ``mean(argmax(logits) == targets)`` over active tokens
            (sparse mode only; ``None`` in dense mode where there is no
            single integer target per row).
    """

    loss: Array
    z_loss: Array
    weight_sum: Array
    accuracy: Array | None


def _flatten_axes(spec_entry) -> list[str]:
    """Return the mesh-axis names referenced by a single PartitionSpec entry."""
    if spec_entry is None:
        return []
    if isinstance(spec_entry, str):
        return [spec_entry]
    if isinstance(spec_entry, tuple):
        out: list[str] = []
        for ax in spec_entry:
            out.extend(_flatten_axes(ax))
        return out
    return []


def _infer_vocab_axis(logits_spec: PartitionSpec | None) -> str | None:
    """Pull the vocab-axis mesh name out of the logits partition spec.

    Returns the last-axis sharding when it's a single string; ``None``
    otherwise (replicated, or sharded over a tuple of axes which the
    caller must handle explicitly).
    """
    if logits_spec is None or len(logits_spec) == 0:
        return None
    last = logits_spec[-1]
    if isinstance(last, str):
        return last
    return None


def _infer_leading_axes(leading_spec: PartitionSpec | None) -> tuple[str, ...]:
    """Return the flat list of mesh axes sharding the leading (batch/seq) dims."""
    if leading_spec is None:
        return ()
    out: list[str] = []
    for entry in leading_spec:
        out.extend(_flatten_axes(entry))
    return tuple(out)


class FusedCrossEntropy(Kernel[FusedCrossEntropyConfig, Array]):
    """Fused cross-entropy with platform + sharding auto-dispatch.

    Computes per-token cross-entropy ``-log p(target | logits)`` together with
    the analytic gradient ``softmax - onehot`` in a single fused kernel.
    The full ``[..., V]`` log-softmax tensor is never materialised in HBM.
    """

    def __init__(self):
        super().__init__(op_id="fused_cross_entropy")

    def get_impl(self, cfg: FusedCrossEntropyConfig):
        platform = detect_platform("fused_cross_entropy", cfg.platform)
        return kernel_registry.get("fused_cross_entropy", platform=platform, backend=cfg.backend)

    def run(
        self,
        logits: Float[Array, "... vocab_size"],
        targets: Int[Array, "..."] | None = None,
        weights: Float[Array, "..."] | None = None,
        *,
        attention_mask: Array | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss: float = 0.0,
        soft_targets: Float[Array, "... vocab_size"] | None = None,
        reduction: str = "mean",
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCrossEntropyConfig,
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Run the registered backend, returning ``(loss, per_row_correct)``.

        ``per_row_correct`` is a 0/1 float array (with sentinel ``-1`` in
        the dense / TP modes where argmax isn't computed) — the public
        wrapper rolls it into :attr:`CrossEntropyOutput.accuracy`.

        When ``attention_mask`` is supplied it is multiplied into
        ``weights`` before dispatch (a position with mask=0 has its loss
        and gradient zeroed out and triggers the kernel's per-block
        sparse early-exit — saving the full ``O(V)`` softmax pass for
        inactive rows). Combining order:
        ``effective_weights = (weights or 1.0) * attention_mask``.
        """
        if attention_mask is not None:
            mask_f32 = attention_mask.astype(jnp.float32)
            if weights is None:
                weights = mask_f32
            else:
                weights = weights.astype(jnp.float32) * mask_f32
        n_rows = 1
        for dim in logits.shape[:-1]:
            n_rows *= int(dim)
        cfg_block_v = int(getattr(cfg, "block_v", self._heuristic_block_v(int(logits.shape[-1]))))
        cfg_block_m = int(getattr(cfg, "block_m", self._heuristic_block_m(n_rows)))
        cfg_num_warps = int(getattr(cfg, "num_warps", 4))
        cfg_num_stages = int(getattr(cfg, "num_stages", 2))
        cfg_backend = getattr(cfg, "backend", Backend.ANY)
        if platform is not None:
            cfg = FusedCrossEntropyConfig(
                block_v=cfg_block_v,
                block_m=cfg_block_m,
                num_warps=cfg_num_warps,
                num_stages=cfg_num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_block_v = cfg.block_v
            cfg_block_m = cfg.block_m
        resolved = detect_platform("fused_cross_entropy", cfg.platform)
        impl = kernel_registry.get("fused_cross_entropy", platform=resolved, backend=cfg.backend)
        return impl(
            logits,
            targets,
            weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            z_loss=z_loss,
            soft_targets=soft_targets,
            reduction=reduction,
            vocab_parallel_axis=vocab_parallel_axis,
            block_v=cfg_block_v,
            block_m=cfg_block_m,
        )

    def create_shard_map_wrapper(
        self,
        logits: Float[Array, "... vocab_size"],
        targets: Int[Array, "..."] | None = None,
        weights: Float[Array, "..."] | None = None,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss: float = 0.0,
        soft_targets: Float[Array, "... vocab_size"] | None = None,
        reduction: str = "mean",
        vocab_parallel_axis: str | None = None,
        platform: PlatformName | None = None,
        cfg: FusedCrossEntropyConfig,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec | None, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = True,
    ):
        """Wrap the loss call in ``shard_map`` with automatic collective insertion.

        Behaviour, deduced from ``in_specs``:

        * The last axis of ``in_specs[0]`` (the logits spec) is treated as
          the vocab-parallel mesh axis. If it's a single string and
          ``vocab_parallel_axis`` was not user-overridden, that mesh axis
          is passed through to ``run`` so the per-shard kernel emits the
          ``pmax`` / ``psum`` collectives needed to merge per-shard
          softmax stats.
        * For ``reduction in ("sum", "mean")``, all mesh axes sharding the
          leading (batch/seq) dimensions of ``targets`` are collected and
          a ``psum`` over them is inserted inside the wrapper so the
          returned scalar is the *global* (mesh-wide) loss.
        * The user-supplied ``check_vma`` is intentionally ignored —
          the collectives inserted inside ``_per_device`` require
          ``check_rep=True`` (shard_map's default) for the gradient to
          flow correctly through the replicated scalar output.

        Returns ``(shard_map_fn, call_args)`` per the Executor contract.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"
        _ = check_vma

        if vocab_parallel_axis is None:
            vocab_parallel_axis = _infer_vocab_axis(in_specs[0])

        targets_spec = in_specs[1] if len(in_specs) > 1 else None
        leading_axes = _infer_leading_axes(targets_spec)

        if weights is None:
            call_args: tuple = (logits, targets)
            actual_in_specs = in_specs[:2]
        else:
            call_args = (logits, targets, weights)
            actual_in_specs = in_specs[:3]
        if len(actual_in_specs) != len(call_args):
            raise ValueError(f"in_specs length {len(actual_in_specs)} != call_args length {len(call_args)}")

        if soft_targets is not None:
            raise NotImplementedError(
                "shard_map dispatch with `soft_targets` is not implemented yet. "
                "Call the kernel directly inside your own shard_map for dense targets."
            )

        _run = self.run
        _ignore_index = ignore_index
        _label_smoothing = label_smoothing
        _z_loss = z_loss
        _reduction = reduction
        _vocab_axis = vocab_parallel_axis
        _platform = platform
        _cfg = cfg
        _leading = leading_axes
        _has_weights = weights is not None
        _inner_red = "none" if reduction == "none" else "sum"

        def _per_device(*args):
            if _has_weights:
                xs, ts, ws = args
            else:
                xs, ts = args
                ws = None
            local_loss, local_correct = _run(
                xs,
                ts,
                ws,
                ignore_index=_ignore_index,
                label_smoothing=_label_smoothing,
                z_loss=_z_loss,
                reduction=_inner_red,
                vocab_parallel_axis=_vocab_axis,
                platform=_platform,
                cfg=_cfg,
            )
            if _reduction == "none":
                return local_loss, local_correct
            if ws is None:
                cnt_local = (ts != _ignore_index).astype(jnp.float32).sum()
            else:
                cnt_local = ws.astype(jnp.float32).sum()
            loss_sum = jax.lax.psum(local_loss, _leading) if _leading else local_loss
            cnt = jax.lax.psum(cnt_local, _leading) if _leading else cnt_local
            if _reduction == "sum":
                final_loss = loss_sum
            else:
                final_loss = loss_sum / jnp.maximum(cnt, 1e-8)
            return final_loss, local_correct

        per_row_spec = targets_spec if targets_spec is not None else PartitionSpec()
        wrapped_out_specs = (out_specs, per_row_spec)

        return (
            shard_map(
                _per_device,
                mesh=mesh,
                in_specs=actual_in_specs,
                out_specs=wrapped_out_specs,
            ),
            call_args,
        )

    @staticmethod
    def _shape_from_inv(inv: Invocation[FusedCrossEntropyConfig, Array]) -> tuple[int, int]:
        """Extract ``(num_rows, vocab_size)`` from the invocation's logits arg."""
        logits = inv.kwargs.get("logits")
        if logits is None and inv.args:
            logits = inv.args[0]
        shape = getattr(logits, "shape", None)
        if shape is None or len(shape) < 2:
            return (0, 0)
        v = int(shape[-1])
        n = 1
        for d in shape[:-1]:
            n *= int(d)
        return (n, v)

    @staticmethod
    def _heuristic_block_v(v: int) -> int:
        """Operation-side ``block_v`` heuristic (mirrors the kernel-side fallback).

        Lives here so the autotuner / heuristic_cfg controls block sizes
        without crossing the operation/kernel boundary.
        """
        if v == 0 or v <= 1024:
            return 256
        if v <= 16384:
            return 512
        if v <= 65536:
            return 1024
        return 2048

    @staticmethod
    def _heuristic_block_m(n: int) -> int:
        return 1 if n < 1024 else 4

    def heuristic_cfg(self, inv: Invocation[FusedCrossEntropyConfig, Array]) -> FusedCrossEntropyConfig:
        n, v = self._shape_from_inv(inv)
        return FusedCrossEntropyConfig(
            block_v=self._heuristic_block_v(v),
            block_m=self._heuristic_block_m(n),
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        return self.candidate_cfgs_gpu(inv)

    def candidate_cfgs_gpu(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        """GPU candidates: enumerate (block_v, block_m, num_warps) for
        tilelang + one XLA baseline.

        The autotuner picks the fastest of these; ``heuristic_cfg`` is
        the cold-start default before autotune results are cached.

        Tuning notes for H100:

        * ``block_v`` ∈ {256, 512, 1024, 2048, 4096, 8192}, pruned by
          actual vocab. Bigger ``block_v`` amortises chunk-loop overhead
          at the cost of SMEM/registers per CTA.
        * ``block_m`` ∈ {1, 2, 4, 8} — ``1`` maximises occupancy for
          wide-vocab; bigger values amortise fixed cost on small-V/big-N.
        * ``num_warps`` ∈ {4, 8} — 8 helps when ``V >= 32K`` (memory-bound).
        * ``num_stages`` ∈ {2, 3} — 3 helps memory-bound large-V.
        * SMEM filter: rough fp32 estimate ``block_v * block_m * 12B``
          must fit in 192KB (H100 envelope; tighter than the 228KB
          hardware limit to leave buffer for the pipeline).
        """
        n, v = self._shape_from_inv(inv)
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[FusedCrossEntropyConfig] = []
        if "tilelang" in platforms:
            block_v_choices: list[int] = []
            for bv in (256, 512, 1024, 2048, 4096, 8192):
                if v == 0 or bv <= max(v, 1):
                    block_v_choices.append(bv)
            if not block_v_choices:
                block_v_choices = [self._heuristic_block_v(v)]
            if n < 1024:
                block_m_choices = [1, 2]
            elif n < 8192:
                block_m_choices = [1, 2, 4]
            else:
                block_m_choices = [1, 4, 8]
            warp_choices = (4, 8) if v >= 32768 else (4,)
            stage_choices = (2, 3) if v >= 16384 else (2,)
            for bv in block_v_choices:
                for bm in block_m_choices:
                    if bv * bm * 4 * 3 > 192 * 1024:
                        continue
                    for warps in warp_choices:
                        for stages in stage_choices:
                            candidates.append(
                                FusedCrossEntropyConfig(
                                    block_v=bv,
                                    block_m=bm,
                                    num_warps=warps,
                                    num_stages=stages,
                                    platform="tilelang",
                                    backend="gpu",
                                )
                            )
        if "xla" in platforms:
            candidates.append(
                FusedCrossEntropyConfig(
                    block_v=0,
                    block_m=0,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[FusedCrossEntropyConfig, Array]):
        return [
            FusedCrossEntropyConfig(
                block_v=0,
                block_m=0,
                num_warps=4,
                num_stages=1,
                platform="xla",
                backend="any",
            )
        ]

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu


_executor: Executor[FusedCrossEntropyConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("fused_cross_entropy"),
    )
)


def fused_cross_entropy(
    logits: Float[Array, "... vocab_size"],
    targets: Int[Array, "..."] | None = None,
    weights: Float[Array, "..."] | None = None,
    /,
    *,
    attention_mask: Array | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets: Float[Array, "... vocab_size"] | None = None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
    platform: PlatformName | None = None,
    cfg: FusedCrossEntropyConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
    check_vma: bool = False,
) -> CrossEntropyOutput:
    """Fused cross-entropy with automatic platform + sharding dispatch.

    Two target modes:
      * **Sparse** (default): integer ``targets`` of shape
        ``logits.shape[:-1]``. Optional ``label_smoothing`` and ``z_loss``
        regularisation fold into the kernel at build time (no runtime
        cost when both are 0).
      * **Dense**: pass ``soft_targets`` (full probability distribution
        over the vocab). ``targets`` is ignored; ``label_smoothing``
        must be applied externally before the call.

    Args:
        logits: ``(..., V)`` predicted logits.
        targets: Integer token ids with shape ``logits.shape[:-1]``.
            Required when ``soft_targets`` is None. Positions equal to
            ``ignore_index`` are excluded from loss and gradient.
        weights: Optional per-token weights of shape ``logits.shape[:-1]``.
        ignore_index: Sparse-mode sentinel for ignored positions.
        label_smoothing: ``α ∈ [0, 1)`` — smoothed target distribution
            ``p[target] = 1 - α``, ``p[v ≠ target] = α / (V - 1)``.
        z_loss: Coefficient for ``z_loss · lse²`` regularisation
            (Mesh-TF / PaLM-style logit magnitude penalty).
        soft_targets: ``(..., V)`` dense probability targets. Switches
            to the dense kernel path.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        vocab_parallel_axis: Mesh axis name along which ``V`` is sharded.
            Usually inferred automatically from ``in_specs``.
        platform: Backend override (``"tilelang"``, ``"xla"``, …).
            Routes through ``kernel_registry``; any registered backend
            name is accepted.
        cfg: Optional :class:`FusedCrossEntropyConfig` override.
        mesh / in_specs / out_specs: When all three are provided the call
            is wrapped in ``jax.shard_map`` automatically.

    Returns:
        :class:`CrossEntropyOutput` NamedTuple with ``(loss, z_loss,
        weight_sum, accuracy)``. ``.loss`` is the differentiable scalar
        (or per-token array for ``reduction="none"``); the other fields
        are detached metrics. For ``jax.grad`` / ``jax.value_and_grad``,
        either index in (``.loss``) or wrap with
        ``lambda *a: fused_cross_entropy(*a).loss``.

    Example (sparse, single-device):
        >>> out = fused_cross_entropy(logits, targets)
        >>> out.loss        # scalar
        >>> out.accuracy    # scalar in [0, 1]

    Example (with label smoothing + z-loss for EasyDeL training):
        >>> out = fused_cross_entropy(
        ...     logits, targets, weights,
        ...     label_smoothing=0.1, z_loss=1e-4,
        ... )
        >>> out.loss, out.z_loss

    Example (distillation — dense soft targets from teacher):
        >>> teacher_probs = jax.nn.softmax(teacher_logits / T, axis=-1)
        >>> out = fused_cross_entropy(
        ...     student_logits, soft_targets=teacher_probs, weights=mask,
        ... )
        >>> out.accuracy is None  # dense mode

    Example (gradient through ``.loss``):
        >>> grads = jax.grad(lambda x: fused_cross_entropy(x, targets).loss)(logits)

    Example (3D mesh, ``dp × sp × tp``):
        Build the mesh once and pass it explicitly along with
        per-input partition specs. The wrapper auto-detects the
        vocab-parallel axis from the **last** entry of ``in_specs[0]``
        (here, ``"tp"``) and inserts ``psum`` over the leading sharded
        axes (``"dp"``, ``"sp"``) so the scalar ``.loss`` is the
        mesh-wide mean.

        >>> from jax.experimental.mesh_utils import create_device_mesh
        >>> from jax.sharding import Mesh, PartitionSpec as P
        >>>
        >>> mesh = Mesh(create_device_mesh((2, 2, 2)), ("dp", "sp", "tp"))
        >>> in_specs = (
        ...     P("dp", "sp", "tp"),  # logits — vocab on tp, batch+seq on dp+sp
        ...     P("dp", "sp"),        # targets — only batch+seq sharded
        ... )
        >>> out = fused_cross_entropy(
        ...     logits, targets,
        ...     mesh=mesh,
        ...     in_specs=in_specs,
        ...     out_specs=P(),       # scalar loss replicated across the mesh
        ... )
        >>> out.loss     # global mean cross-entropy
        >>> out.accuracy # global accuracy

        With per-token ``weights``, extend ``in_specs`` to three entries:

        >>> in_specs = (
        ...     P("dp", "sp", "tp"),
        ...     P("dp", "sp"),
        ...     P("dp", "sp"),       # weights — same sharding as targets
        ... )
        >>> out = fused_cross_entropy(
        ...     logits, targets, weights,
        ...     mesh=mesh, in_specs=in_specs, out_specs=P(),
        ... )
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    loss, per_row_correct = _executor(
        FusedCrossEntropy(),
        logits=logits,
        targets=targets,
        weights=weights,
        attention_mask=attention_mask,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        soft_targets=soft_targets,
        reduction=reduction,
        vocab_parallel_axis=vocab_parallel_axis,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
        _cfg=cfg,
    )

    if weights is not None:
        flat_w = weights.reshape(-1).astype(jnp.float32)
    elif soft_targets is None and targets is not None:
        flat_w = (targets.reshape(-1) != ignore_index).astype(jnp.float32)
    else:
        flat_w = jnp.ones(logits.shape[:-1], dtype=jnp.float32).reshape(-1)
    if attention_mask is not None:
        flat_w = flat_w * attention_mask.reshape(-1).astype(jnp.float32)
    weight_sum = jax.lax.stop_gradient(flat_w.sum())

    if z_loss > 0.0:
        lse = jax.scipy.special.logsumexp(logits, axis=-1)
        per_row_zterm = z_loss * lse * lse
        if reduction == "none":
            z_loss_metric = per_row_zterm.astype(jnp.float32)
        elif reduction == "sum":
            z_loss_metric = jnp.sum(per_row_zterm * flat_w.reshape(logits.shape[:-1]))
        else:
            z_loss_metric = jnp.sum(per_row_zterm * flat_w.reshape(logits.shape[:-1])) / jnp.maximum(weight_sum, 1e-8)
        z_loss_metric = jax.lax.stop_gradient(z_loss_metric)
    else:
        z_loss_metric = jnp.zeros((), dtype=jnp.float32)

    accuracy: Array | None
    is_per_row_sentinel = soft_targets is not None or targets is None or vocab_parallel_axis is not None
    if is_per_row_sentinel:
        accuracy = None
    else:
        per_row_correct = jax.lax.stop_gradient(per_row_correct)
        if reduction == "none":
            accuracy = per_row_correct
        else:
            num_correct = jnp.sum(per_row_correct)
            num_active = jnp.maximum(flat_w.sum(), 1e-8)
            accuracy = num_correct / num_active

    return CrossEntropyOutput(
        loss=loss,
        z_loss=z_loss_metric,
        weight_sum=weight_sum,
        accuracy=accuracy,
    )
