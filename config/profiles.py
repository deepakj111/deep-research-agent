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
from pydantic import BaseModel, ConfigDict

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    max_web_results: int
    max_arxiv_papers: int
    max_github_repos: int
    query_decomposition: str


@functools.lru_cache(maxsize=16)
def load_profile(name: str) -> dict[str, Any]:
    """Load a research profile YAML by name, validate it, and return as dict."""
    with open(_PROFILES_DIR / f"{name}.yaml") as f:
        raw = yaml.safe_load(f)

    # Validate against schema to catch typos early
    validated = ProfileConfig(**raw)
    return validated.model_dump()
