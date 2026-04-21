"""
agent/middleware/pii_filter.py

Regex-based PII redaction middleware.

Applied to all MCP tool outputs before they enter the LangGraph state.
This prevents personally identifiable information from leaking into
research reports — a critical requirement for enterprise AI deployments.

Redaction is conservative: we match well-known PII patterns (SSN, credit
card numbers, emails, phone numbers, IP addresses) and replace them with
bracketed placeholders. A redaction count is returned for audit logging.

Design notes:
  - Patterns are compiled once at module load for performance.
  - filter_pii() is a pure function with no side effects — easy to test.
  - The audit dict lets the observability layer record how much PII was
    scrubbed per tool call without storing the actual PII.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─────────────────────── Compiled PII Patterns ───────────────────────────────

_PII_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # (compiled regex, replacement text, pattern name for audit)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN REDACTED]",
        "ssn",
    ),
    (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[CARD REDACTED]",
        "credit_card",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL REDACTED]",
        "email",
    ),
    (
        re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
        "[PHONE REDACTED]",
        "phone",
    ),
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[IP REDACTED]",
        "ip_address",
    ),
]


# ─────────────────────── Public API ──────────────────────────────────────────


@dataclass
class RedactionResult:
    """Result of a PII filtering pass."""

    text: str
    total_redactions: int = 0
    redaction_counts: dict[str, int] = field(default_factory=dict)


def filter_pii(text: str) -> RedactionResult:
    """
    Scan text for PII patterns and replace matches with safe placeholders.

    Args:
        text: Raw text from an MCP tool response.

    Returns:
        RedactionResult with cleaned text and an audit trail of what was redacted.
    """
    total = 0
    counts: dict[str, int] = {}

    for pattern, replacement, name in _PII_RULES:
        matches = pattern.findall(text)
        if matches:
            count = len(matches)
            counts[name] = count
            total += count
            text = pattern.sub(replacement, text)

    return RedactionResult(text=text, total_redactions=total, redaction_counts=counts)


def filter_pii_simple(text: str) -> str:
    """
    Convenience wrapper that returns only the cleaned text.

    Use this in agent nodes where you don't need the audit trail.
    """
    return filter_pii(text).text
