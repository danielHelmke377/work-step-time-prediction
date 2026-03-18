"""
conftest.py — pytest configuration

Adds ``src/`` to ``sys.path`` so that the ``repair_order`` package is
importable by editors, type-checkers, and testing tools that do not
activate the virtual environment automatically.

The canonical installation is::

    pip install -e .[dev]

which makes ``repair_order`` importable without this shim. This file is
retained as a backstop for CI runners and IDE terminals that bypass the
editable install.
"""
import sys
from pathlib import Path

# Prepend src/ so repair_order is importable without a full editable install
sys.path.insert(0, str(Path(__file__).parent / "src"))
