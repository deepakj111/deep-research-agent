"""
agent/middleware/__init__.py

Agent middleware components for input/output sanitization.
"""

from agent.middleware.pii_filter import filter_pii

__all__ = ["filter_pii"]
