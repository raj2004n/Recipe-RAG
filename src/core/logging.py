import logging
import os
import sys


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DEFAULT_LEVEL = os.environ.get("RECIPE_RAG_LOG_LEVEL", "INFO").upper()


_configured = False


def configure_logging(
    level: str | int | None = None,
    stream=None,
) -> None:
    """Configure the root logger with a single stderr stream handler.

    Args:
        level: logging level name or numeric value. Defaults to
            `RECIPE_RAG_LOG_LEVEL` env var, then `INFO`.
        stream: target stream. Defaults to `sys.stderr`.
    """
    global _configured
    if _configured:
        return

    if level is None:
        level = _DEFAULT_LEVEL

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    _configured = True
