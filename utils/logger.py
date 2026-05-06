"""
utils/logger.py

Centralized structured logging for the DeepResearch Agent.

Features:
  - JSON or plain-text format controlled by LOG_FORMAT env var.
  - Automatic run_id injection via a logging.Filter that reads from
    contextvars — no manual threading of run_id through call stacks.
  - Suppresses verbose third-party loggers (httpx, urllib3).

JSON mode is intended for production deployments with log aggregation
(Datadog, Grafana Loki, CloudWatch). Plain-text mode is the default for
local development.
"""

import logging
import logging.config
import os
import sys

from utils.context import get_run_id


class RunIdFilter(logging.Filter):
    """Inject the current run_id into every log record.

    The run_id is read from the contextvars-based context set by
    ``utils.context.bind_run_id()``. If no run_id is bound (e.g.
    during startup), the field defaults to an empty string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()  # type: ignore[attr-defined]
        return True


def setup_logging() -> None:
    """
    Configure centralized structured logging for the deep-research-agent.

    Set ``LOG_FORMAT=json`` in the environment for JSON-structured output
    suitable for log aggregation. Defaults to human-readable plain text.
    """
    log_format = os.environ.get("LOG_FORMAT", "standard").lower()

    # Choose formatter based on environment
    formatter_key = "json" if log_format == "json" else "standard"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "run_id": {
                "()": lambda: RunIdFilter(),
            },
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(run_id)s]: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                "defaults": {"run_id": ""},
            },
            "json": {
                "()": "logging.Formatter",
                "format": (
                    '{"time": "%(asctime)s", "level": "%(levelname)s", '
                    '"name": "%(name)s", "run_id": "%(run_id)s", '
                    '"message": "%(message)s"}'
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                "defaults": {"run_id": ""},
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": formatter_key,
                "filters": ["run_id"],
                "stream": sys.stdout,
                "level": "INFO",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "loggers": {
            "agent": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "api": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            # Suppress overly verbose third-party logs
            "httpx": {
                "level": "WARNING",
            },
            "urllib3": {
                "level": "WARNING",
            },
        },
    }
    logging.config.dictConfig(logging_config)
