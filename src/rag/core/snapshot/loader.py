"""Snapshot loading.

A 'Snapshot' is the read-only artefact Pipelines consume at inference time.
Each strategy declares which index loaders to run, so adding a strategy is a
matter of adding a row to `_STRATEGY_LOADERS`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ...retrieval.dense.indexer import IndexedChunkSet
from ...retrieval.sparse.bm25indexer import BM25IndexedChunkSet
from .indices import load_bm25_index_chunk_sets, load_indexed_chunk_sets


@dataclass
class Snapshot:
    snapshot_id: str | None
    created_at: str | None
    retrieval_strategy: str
    indexed_chunk_sets: list[IndexedChunkSet]  # carries chunks in IndexedChunkSet.chunks
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet]  # carries chunks in BM25IndexedChunkSet.chunks
    base_config: dict[str, any]  # configs for initialising retriever components for inference, in pipeline
    prompts_config: dict[str, any]


# Strategy -> ordered list of index loaders. Each loader returns
# (chunk_sets, chunks_by_set_name). Adding a strategy = add a row.
_STRATEGY_LOADERS = {
    "dense":  [load_indexed_chunk_sets],
    "hybrid": [load_indexed_chunk_sets, load_bm25_index_chunk_sets],
}


def load_snapshot(path: str) -> Snapshot:
    path_obj = Path(path)

    # load configs
    with open(path_obj / "configs" / "base_config.yaml") as f:
        base_config = yaml.safe_load(f)
    with open(path_obj / "configs" / "prompts_config.yaml") as f:
        prompts_config = yaml.safe_load(f)

    # load metadata first to determine strategy
    with open(path_obj / "metadata.json") as f:
        metadata = json.load(f)

    strategy = metadata["strategy"]
    try:
        loaders = _STRATEGY_LOADERS[strategy]
    except KeyError as e:
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}; "
            f"expected one of {sorted(_STRATEGY_LOADERS)}"
        ) from e

    indexed_chunk_sets: list[IndexedChunkSet] = []
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet] = []

    for loader in loaders:
        chunk_sets, chunks_by_set_name = loader(
            path_obj / "index",
            chunks_save_path=path_obj / "chunks",
        )
        if loader is load_bm25_index_chunk_sets:
            bm25_indexed_chunk_sets = chunk_sets
            for indexed in bm25_indexed_chunk_sets:
                indexed.chunks = chunks_by_set_name.get(indexed.name, {})
        else:
            indexed_chunk_sets = chunk_sets
            for indexed in indexed_chunk_sets:
                indexed.chunks = chunks_by_set_name.get(indexed.name, {})

    return Snapshot(
        snapshot_id=metadata["snapshot_id"],
        created_at=metadata["created_at"],
        retrieval_strategy=strategy,
        indexed_chunk_sets=indexed_chunk_sets,
        bm25_indexed_chunk_sets=bm25_indexed_chunk_sets,
        base_config=base_config,
        prompts_config=prompts_config,
    )
