"""Snapshot subsystem.

Public surface is split across focused submodules:
    - paths       : SnapshotPaths dataclass
    - chunk_sets  : ChunkSet / EncodedChunkSet / TokenizedChunkSet IO
    - indices     : FAISS and BM25 index IO
    - lifecycle   : finalise / archive / config-copy
    - loader      : Snapshot dataclass + load_snapshot

Import names from the submodule that owns them, e.g.
    from src.rag.core.snapshot.loader import load_snapshot, Snapshot
    from src.rag.core.snapshot.chunk_sets import save_chunk_sets
"""

from .loader import Snapshot, load_snapshot

__all__ = ["Snapshot", "load_snapshot"]
