"""Structured logging configuration with structlog.

CLAUDE.md rule §10: never use ``print()`` in production — always
``structlog.get_logger()``. This configuration produces JSON in staging/prod and
a readable console in dev.
"""

import logging
import sys

import structlog


def configure_logging(env: str, log_level: str) -> None:
    """Configure structlog according to the environment.

    Args:
        env: ``dev``, ``staging`` or ``prod``.
        log_level: Root level, e.g. ``INFO``, ``DEBUG``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if env == "dev":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
