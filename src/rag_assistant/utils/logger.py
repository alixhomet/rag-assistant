"""Configuration centralisée du logging."""

import logging
import sys

from rag_assistant.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure le logger racine du projet (une seule fois)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger("rag_assistant")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé sous l'espace 'rag_assistant'."""
    configure_logging()
    return logging.getLogger(f"rag_assistant.{name}")
