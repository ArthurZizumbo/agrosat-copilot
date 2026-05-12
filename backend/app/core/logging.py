"""Configuración de logging estructurado con structlog.

Regla §10 de CLAUDE.md: nunca usar ``print()`` en producción — siempre
``structlog.get_logger()``. Esta configuración produce JSON en staging/prod
y consola legible en dev.
"""

import logging
import sys

import structlog


def configure_logging(env: str, log_level: str) -> None:
    """Configura structlog según el entorno.

    Args:
        env: ``dev``, ``staging`` o ``prod``.
        log_level: Nivel raíz, p. ej. ``INFO``, ``DEBUG``.
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
