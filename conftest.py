"""
conftest.py — pytest configuration
Adds the src/ directory to sys.path so that `repair_order` is importable
without requiring `pip install -e .`
"""
import sys
from pathlib import Path

# Insert src/ at the front of sys.path so tests can import repair_order
sys.path.insert(0, str(Path(__file__).parent / "src"))
