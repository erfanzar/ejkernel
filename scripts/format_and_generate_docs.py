#!/usr/bin/env python3
"""
Standalone script to format code and generate API documentation for ejkernel.
This works as a pre-commit hook when called from .pre-commit-config.yaml.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from scripts/
PROJECT_NAME = "ejkernel"
DOCS_API_DIR = PROJECT_ROOT / "docs" / "api_docs"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command and arguments as list.
        check: Whether to raise exception on non-zero exit code.

    Returns:
        CompletedProcess object with command results.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def format_code(directory: str = PROJECT_NAME, fix: bool = True) -> bool:
    """
    Format Python code using ruff.

    Args:
        directory: Directory to format.
        fix: Whether to apply fixes automatically.

    Returns:
        True if successful, False otherwise.
    """
    print(f"Formatting code in {directory}/...")

    # Get all Python files
    python_files = list(Path(directory).rglob("*.py"))

    if not python_files:
        print("No Python files found.")
        return True

    success = True

    # Run ruff check with optional fixes
    if fix:
        print("Running ruff check with fixes...")
        result = run_command(
            ["ruff", "check", "--fix", "--config", "pyproject.toml"] + [str(f) for f in python_files], check=False
        )
        if result.returncode != 0:
            print(f"Ruff check found issues (exit code: {result.returncode})")
            success = False

    # Run ruff format
    print("Running ruff format...")
    result = run_command(["ruff", "format", "--config", "pyproject.toml"] + [str(f) for f in python_files], check=False)
    if result.returncode != 0:
        print(f"Ruff format failed (exit code: {result.returncode})")
        success = False

    if success:
        print(f"✓ Successfully formatted {len(python_files)} files")
    else:
        print("✗ Some files had formatting issues")

    return success


def discover_modules(base_dir: str = PROJECT_NAME) -> dict[tuple[str, ...], str]:
    """
    Discover all Python modules in the project.

    Args:
        base_dir: Base directory to search.

    Returns:
        Dictionary mapping module paths to import names.
    """
    modules = {}
    base_path = Path(base_dir)

    for py_file in base_path.rglob("*.py"):
        # Skip test files and private modules
        if any(part.startswith("test") for part in py_file.parts):
            continue
        if "__pycache__" in str(py_file):
            continue

        # Convert to module path
        relative = py_file.relative_to(PROJECT_ROOT)
        module_parts = list(relative.parts[:-1])  # Remove .py file
        if relative.stem != "__init__":
            module_parts.append(relative.stem)

        # Create display name
        display_parts = tuple(" ".join(word.capitalize() for word in part.split("_")) for part in module_parts)

        # Create import path
        import_path = ".".join(module_parts)
        modules[display_parts] = import_path

    return modules


def create_rst_file(name: str, module_path: str, output_dir: Path) -> None:
    """
    Create an RST documentation file for a module.

    Args:
        name: Display name for the module.
        module_path: Import path for the module.
        output_dir: Directory to write RST files.
    """
    # Create filename from module path
    filename = module_path.replace(f"{PROJECT_NAME}.", "").replace(".", "_") + ".rst"
    rst_path = output_dir / filename

    # Write RST content
    title = name
    with open(rst_path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f".. automodule:: {module_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")


def generate_api_docs(clean: bool = True) -> bool:
    """
    Generate API documentation RST files.

    Args:
        clean: Whether to clean existing docs first.

    Returns:
        True if successful, False otherwise.
    """
    print("Generating API documentation...")

    # Ensure docs directory exists
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing docs if requested
    if clean:
        for rst_file in DOCS_API_DIR.glob("*.rst"):
            rst_file.unlink()

    # Discover modules
    modules = discover_modules(PROJECT_NAME)

    if not modules:
        print("No modules found to document")
        return False

    # Generate RST files
    for display_parts, module_path in modules.items():
        name = display_parts[-1] if display_parts else "Index"
        create_rst_file(name, module_path, DOCS_API_DIR)

    # Generate index file
    index_path = DOCS_API_DIR / "index.rst"
    with open(index_path, "w") as f:
        f.write(f"{PROJECT_NAME.upper()} API Reference\n")
        f.write("=" * (len(PROJECT_NAME) + 14) + "\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")

        # Add all generated RST files
        for rst_file in sorted(DOCS_API_DIR.glob("*.rst")):
            if rst_file.name != "index.rst":
                f.write(f"   {rst_file.stem}\n")

    print(f"✓ Generated documentation for {len(modules)} modules")
    return True


def run_tests(test_dir: str = "test") -> bool:
    """
    Run project tests using pytest.

    Args:
        test_dir: Directory containing tests.

    Returns:
        True if tests pass, False otherwise.
    """
    print(f"Running tests in {test_dir}/...")

    result = run_command(["pytest", test_dir, "-v"], check=False)

    if result.returncode == 0:
        print("✓ All tests passed")
        return True
    else:
        print("✗ Some tests failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=f"Format code and generate documentation for {PROJECT_NAME}")

    # Task selection
    parser.add_argument("--format", action="store_true", help="Format code with ruff")
    parser.add_argument("--docs", action="store_true", help="Generate API documentation")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--all", action="store_true", help="Run all tasks")

    # Options
    parser.add_argument(
        "--no-fix", dest="fix", action="store_false", default=True, help="Don't apply fixes automatically"
    )
    parser.add_argument(
        "--no-clean", dest="clean", action="store_false", default=True, help="Don't clean old documentation"
    )
    parser.add_argument("--directory", default=PROJECT_NAME, help=f"Directory to format (default: {PROJECT_NAME})")

    args = parser.parse_args()

    # Default to all if no specific task selected
    if not any([args.format, args.docs, args.test, args.all]):
        args.all = True

    exit_code = 0

    # Run selected tasks
    if args.all or args.format:
        if not format_code(args.directory, fix=args.fix):
            exit_code = 1

    if args.all or args.docs:
        if not generate_api_docs(clean=args.clean):
            exit_code = 1

    if args.test:
        if not run_tests():
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
