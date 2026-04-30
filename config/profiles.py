# config/profiles.py
"""
Centralized YAML profile loader.

All agent nodes and the planner share this single cached loader
instead of each maintaining their own copy.
"""

from __future__ import annotations

import functools
from typing import Any

import yaml


@functools.lru_cache(maxsize=16)
def load_profile(name: str) -> dict[str, Any]:
    """Load a research profile YAML by name, with in-process caching."""
    with open(f"config/profiles/{name}.yaml") as f:
        return yaml.safe_load(f)
