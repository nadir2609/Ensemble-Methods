"""Pytest configuration: ensure the project root is importable as ``src.*``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
