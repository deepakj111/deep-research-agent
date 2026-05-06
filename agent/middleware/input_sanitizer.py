"""
agent/middleware/input_sanitizer.py

Regex-based prompt injection defense for user-supplied research queries.

Applied at the classifier node — the first LLM-touching node in the graph —
so that no downstream node ever sees a raw, unsanitized query.

Design:
  - Each rule is a compiled regex with a human-readable name for audit logging.
  - Dangerous patterns are stripped (not blocked) so legitimate queries that
    accidentally contain trigger phrases still succeed.
  - A sanitization report is returned for the thought_log so the run record
    shows what was scrubbed and why.
  - The sanitizer is intentionally conservative: it targets well-known
    injection vectors without being so aggressive that it mangles normal
    research queries.

References:
  - OWASP LLM Top 10 (2025): LLM01 — Prompt Injection
  - Simon Willison's prompt injection taxonomy
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────── Compiled Injection Patterns ─────────────────────────

_INJECTION_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # (compiled regex, replacement text, pattern name for audit)
    # Direct instruction overrides
    (
        re.compile(
            r"(?i)(ignore|disregard|forget|override|bypass)\s+"
            r"(all\s+)?(previous|prior|above|earlier|preceding)\s+"
            r"(instructions?|prompts?|rules?|context|guidelines?|directions?)",
        ),
        "[INJECTION REMOVED]",
        "instruction_override",
    ),
    # Role/identity hijacking
    (
        re.compile(
            r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|"
            r"assume\s+the\s+role|switch\s+to\s+role|new\s+persona)",
        ),
        "[INJECTION REMOVED]",
        "role_hijack",
    ),
    # System prompt extraction
    (
        re.compile(
            r"(?i)(output|print|reveal|show|display|repeat|echo)\s+"
            r"(your\s+)?(system\s+prompt|initial\s+instructions?|"
            r"original\s+prompt|hidden\s+instructions?|system\s+message)",
        ),
        "[INJECTION REMOVED]",
        "prompt_extraction",
    ),
    # Delimiter-based injection (markdown/XML fence breaking)
    (
        re.compile(
            r"(?i)(```\s*system|<\|?\s*system\s*\|?>|<<\s*SYS\s*>>|"
            r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>)",
        ),
        "[INJECTION REMOVED]",
        "delimiter_injection",
    ),
    # "Do anything now" / DAN-style jailbreaks
    (
        re.compile(
            r"(?i)(DAN\s+mode|do\s+anything\s+now|jailbreak|"
            r"developer\s+mode\s+enabled|evil\s+mode|unleash)",
        ),
        "[INJECTION REMOVED]",
        "jailbreak_attempt",
    ),
]


# ─────────────────────── Public API ──────────────────────────────────────────


@dataclass
class SanitizationResult:
    """Result of a prompt injection scan."""

    text: str
    total_detections: int = 0
    detection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        return self.total_detections > 0


def sanitize_query(text: str) -> SanitizationResult:
    """
    Scan a user query for prompt injection patterns and neutralize them.

    Args:
        text: Raw user query string.

    Returns:
        SanitizationResult with cleaned text and an audit trail.
    """
    total = 0
    counts: dict[str, int] = {}

    for pattern, replacement, name in _INJECTION_RULES:
        matches = pattern.findall(text)
        if matches:
            count = len(matches)
            counts[name] = count
            total += count
            text = pattern.sub(replacement, text)

    if total > 0:
        logger.warning(
            "[InputSanitizer] Detected %d injection pattern(s): %s",
            total,
            counts,
        )

    return SanitizationResult(text=text, total_detections=total, detection_counts=counts)
