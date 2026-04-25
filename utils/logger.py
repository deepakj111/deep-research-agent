import logging
import logging.config
import sys


def setup_logging() -> None:
    """
    Configures centralized structured logging for the deep-research-agent.
    Ensures that all application modules log consistently.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "json": {
                "()": "logging.Formatter",
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
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
