# config/profiles.py
"""
Centralized YAML profile loader.

All agent nodes and the planner share this single cached loader
instead of each maintaining their own copy.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


@functools.lru_cache(maxsize=16)
def load_profile(name: str) -> dict[str, Any]:
    """Load a research profile YAML by name, with in-process caching."""
    with open(_PROFILES_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)
