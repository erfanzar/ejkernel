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


"""Execution engine and batch processing for ejkernel.ops.

This module provides the main execution engine for running kernels with automatic
configuration selection, as well as utilities for batch processing, autotuning,
and profiling.

Classes:
    Executor: Main execution engine coordinating config selection and kernel execution
    ConfigChooser: Protocol for configuration selection strategies
    Autotuner: Core autotuner with measurement collection and basic timing
    FNAutotuner: Advanced class-based autotuner with profiler integration
    AutotuneData: Container for autotuning measurements and results
    AutotuningResult: Results from autotuning recorded operations
    Entry: Single autotuning result entry
    Measurement: Single performance measurement

Functions:
    vmap_with_config: Vectorized execution with shared configuration selection
    pmap_with_config: Parallel execution across devices
    autotune: Decorator for automatic hyperparameter optimization
    autotune_recorded: Autotune all recorded operations
    autotune_lowered: Autotune operations found in lowered computations
    benchmark: Simple function benchmarking utility

Example:
    >>> from ejkernel.ops.execution import Executor, autotune
    >>>
    >>> # Using the executor for kernel execution
    >>> executor = Executor(selector)
    >>> result = executor(my_kernel, input_data)
    >>>
    >>> # Using autotune decorator for hyperparameter optimization
    >>> @autotune(hyperparams={'block_size': [64, 128, 256]})
    ... def my_function(x, block_size=128):
    ...     return process(x, block_size)
"""

from .batch import pmap_with_config, vmap_with_config
from .executor import ConfigChooser, Executor
from .offline import autotune_lowered
from .tuning import (
    AutotuneData,
    Autotuner,
    AutotuningResult,
    Entry,
    FNAutotuner,
    Measurement,
    autotune,
    autotune_recorded,
    benchmark,
)

# Alias for backward compatibility and convenience
Tuner = Autotuner

__all__ = (
    "AutotuneData",
    "Autotuner",
    "AutotuningResult",
    "ConfigChooser",
    "Entry",
    "Executor",
    "FNAutotuner",
    "Measurement",
    "Tuner",
    "autotune",
    "autotune_lowered",
    "autotune_recorded",
    "benchmark",
    "pmap_with_config",
    "vmap_with_config",
)
