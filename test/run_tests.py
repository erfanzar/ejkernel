#!/usr/bin/env python3
"""Test runner for ejkernels test suite.

Usage:
    python test/run_tests.py                    # Run all tests
    python test/run_tests.py --xla              # Run only XLA tests
    python test/run_tests.py --triton           # Run only Triton tests
    python test/run_tests.py --comparison       # Run only comparison tests
    python test/run_tests.py --verbose          # Verbose output
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

    # Build pytest arguments
    pytest_args = []

    # Determine which tests to run
    if args.xla:
        pytest_args.append("test/kernels/_xla")
    elif args.triton:
        pytest_args.append("test/kernels/_triton")
    elif args.comparison:
        pytest_args.append("test/kernels/comparison")
    else:
        # Run all tests
        pytest_args.append("test/kernels")

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-v")  # Always verbose for better output

    # Add keyword filter
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Add fail fast
    if args.failfast:
        pytest_args.append("-x")

    # Traceback style
    pytest_args.append(f"--tb={args.tb}")

    # Add test summary
    pytest_args.append("-ra")

    print("=" * 70)
    print("Running ejkernels test suite")
    print("=" * 70)
    print(f"Test path: {pytest_args[0]}")
    print(f"Options: {' '.join(pytest_args[1:])}")
    print("=" * 70)
    print()

    # Run pytest
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
