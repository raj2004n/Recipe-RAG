from typing import Any

_LOADER_NAMES = frozenset({
    "Snapshot",
    "load_snapshot",
    "load_snapshot_by_target",
})

__all__ = [
    "Snapshot",
    "load_snapshot",
    "load_snapshot_by_target",
]


def __getattr__(name: str) -> Any:
    """Bind one of the loader's names the first time a caller asks for it."""
    if name in _LOADER_NAMES:
        from . import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
