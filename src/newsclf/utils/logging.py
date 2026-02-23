from __future__ import annotations

import logging
import os


def setup_logging(name: str = "newsclf", level: str | None = None) -> logging.Logger:
    """
    Simple stdout logging with a consistent format.
    Level priority:
      1) explicit 'level'
      2) env var LOG_LEVEL
      3) INFO
    """
    lvl = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(lvl)

    # Avoid duplicate handlers if called multiple times (common in tests)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.propagate = False
    return logger