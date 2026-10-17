from typing import Any

from .manifest import add_to_dvc, write_manifest

_LAZY_NAMES = frozenset({"train_student"})

__all__ = ["write_manifest", "add_to_dvc", "train_student"]


def __getattr__(name: str) -> Any:
    """Bind the training entry point on first use.

    The training module reads back through the runtime package, which reads the
    prompt path, which reads this one, so binding it here would close that loop
    at import time. Staying lazy also spares a caller of `write_manifest` the
    torch import that training carries.
    """
    if name in _LAZY_NAMES:
        from . import student
        return getattr(student, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
