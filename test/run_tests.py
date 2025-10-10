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


"""Test runner for ejkernels test suite.

Usage:
    python test/run_tests.py
    python test/run_tests.py --xla
    python test/run_tests.py --triton
    python test/run_tests.py --comparison
    python test/run_tests.py --verbose
"""

import argparse
import sys

import pytest


def main():
    """Run test suite with specified options."""
    parser = argparse.ArgumentParser(description="Run ejkernels test suite")
    parser.add_argument("--xla", action="store_true", help="Run only XLA kernel tests")
    parser.add_argument("--triton", action="store_true", help="Run only Triton kernel tests")
    parser.add_argument("--comparison", action="store_true", help="Run only comparison tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-k", "--keyword", type=str, help="Only run tests matching keyword")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    parser.add_argument("--tb", type=str, default="short", help="Traceback style (short/long/native)")

    args = parser.parse_args()

    pytest_args = []

    if args.xla:
        pytest_args.append("test/kernels/_xla")
    elif args.triton:
        pytest_args.append("test/kernels/_triton")
    elif args.comparison:
        pytest_args.append("test/kernels/comparison")
    else:
        pytest_args.append("test/kernels")

    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-v")

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    if args.failfast:
        pytest_args.append("-x")

    pytest_args.append(f"--tb={args.tb}")

    pytest_args.append("-ra")

    print("=" * 70)
    print("Running ejkernels test suite")
    print("=" * 70)
    print(f"Test path: {pytest_args[0]}")
    print(f"Options: {' '.join(pytest_args[1:])}")
    print("=" * 70)
    print()

    exit_code = pytest.main(pytest_args)

    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
