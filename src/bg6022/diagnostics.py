"""Application-owned logging policy for noisy scientific dependencies."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_KNOWN_H_WARNING = re.compile(
    r"^(?:\[\d{2}:\d{2}:\d{2}\]\s*)?"
    r"WARNING: not removing hydrogen atom without neighbors\s*$"
)


class _RdkitConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.WARNING
            and _KNOWN_H_WARNING.fullmatch(record.getMessage()) is not None
        )


def configure_rdkit_logging(data_root: str | Path) -> None:
    """Keep RDKit diagnostics in a bounded file and filter one known warning."""

    from rdkit import rdBase

    log_path = Path(data_root).resolve() / "logs" / "rdkit.log"
    logger = logging.getLogger("rdkit")
    if any(
        getattr(handler, "_bg6022_log_path", None) == str(log_path) for handler in logger.handlers
    ):
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    file_handler._bg6022_log_path = str(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.addFilter(_RdkitConsoleFilter())

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    rdBase.LogToPythonLogger()


__all__ = ["configure_rdkit_logging"]
