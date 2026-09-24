"""Snapshot subsystem.

Public surface is split across focused submodules:
    - paths       : SnapshotPaths dataclass
    - chunk_sets  : ChunkSet / EncodedChunkSet / TokenizedChunkSet IO
    - indices     : FAISS and BM25 index IO
    - lifecycle   : start_new_build / finalise / config-copy
    - registry    : naming + registry.json maintenance
    - loader      : Snapshot dataclass + load_snapshot + load_snapshot_by_target

Import names from the submodule that owns them, e.g.
    from src.rag.core.snapshot.loader import Snapshot, load_snapshot, load_snapshot_by_target
    from src.rag.core.snapshot.chunk_sets import save_chunk_sets

The loader's names are also reachable from the package itself, which is how the
pipelines take them, but they are bound on first access rather than at import.
The loader reads chunk classes out of the retrieval stack, so importing it loads
the encoder libraries with it, and a caller that only wants the registry or the
paths should not pay for that. Asking for `Snapshot` or `load_snapshot` still
works exactly as it did, and binds the loader at that moment.
"""

from typing import Any

_LOADER_NAMES = frozenset({
    "Snapshot",
    "load_snapshot",
    "load_snapshot_by_target",
    "load_current_snapshot",
})

__all__ = [
    "Snapshot",
    "load_snapshot",
    "load_snapshot_by_target",
    "load_current_snapshot",
]


def __getattr__(name: str) -> Any:
    """Bind one of the loader's names the first time a caller asks for it."""
    if name in _LOADER_NAMES:
        from . import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
