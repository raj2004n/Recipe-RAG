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
"""

from .loader import Snapshot, load_snapshot, load_snapshot_by_target, load_current_snapshot

__all__ = [
    "Snapshot",
    "load_snapshot",
    "load_snapshot_by_target",
    "load_current_snapshot",
]
