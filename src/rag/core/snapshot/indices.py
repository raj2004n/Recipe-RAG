import json
from pathlib import Path

import faiss

from ...retrieval.chunker import Chunk
from ...retrieval.dense.indexer import IndexedChunkSet
from ...retrieval.sparse.bm25indexer import BM25IndexedChunkSet
from ...retrieval.sparse.tokenizer import TokenizedChunk
from .chunk_sets import load_chunk_sets


def save_indexed_chunk_sets(indexed_chunk_sets: list[IndexedChunkSet], save_path: str):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i, chunk_set in enumerate(indexed_chunk_sets):
        index_filename = f"index_{i}.faiss"

        faiss.write_index(
            chunk_set.index,
            str(save_path / index_filename),
        )

        manifest.append({
            "name": chunk_set.name,
            "index": index_filename,
        })

    with open(save_path / "indexed_chunk_sets.json", "w") as f:
        json.dump(manifest, f, indent=2)


def save_bm25_index_chunk_sets(bm25_index_chunk_sets: list[BM25IndexedChunkSet], save_path: str):
    """ Save BM25-indexed chunk sets.

    The BM25 index is reconstructed on load from the serialized tokenized chunks,
    so no raw index file is needed - only the manifest and tokenized data.
    """
    from dataclasses import asdict

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    manifest = []

    for chunk_set in bm25_index_chunk_sets:
        # store tokenized chunks inline in manifest - BM25 index is rebuilt on load
        manifest.append({
            "name": chunk_set.name,
            "tokenized_chunks": [
                {
                    "chunk": asdict(tc.chunk),
                    "tokens": tc.tokens,
                }
                for tc in chunk_set.tokenized_chunks
            ],
        })

    with open(save_path / "bm25_index_chunk_sets.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _load_chunks_by_set_name(
    chunks_save_path: Path | None,
) -> dict[str, dict[int, Chunk]]:
    """Group loaded chunks by chunk-set name and inner-dict by chunk_id.

    Returns an empty dict if `chunks_save_path` is None.
    """
    if chunks_save_path is None:
        return {}
    return {
        cs.name: {c.chunk_id: c for c in cs.chunks}
        for cs in load_chunk_sets(str(chunks_save_path))
    }


def load_bm25_index_chunk_sets(
    save_path: str,
    chunks_save_path: str | None = None,
) -> tuple[list[BM25IndexedChunkSet], dict[str, dict[int, Chunk]]]:
    """ Load BM25-indexed chunk sets.

    Args:
        save_path: directory holding the bm25_index_chunk_sets.json manifest.
        chunks_save_path: directory holding chunk_sets.json. If provided,
            also load chunks and group them by set name so the caller can
            attach them to the returned BM25IndexedChunkSets.

    Returns:
        (bm25_index_chunk_sets, chunks_by_set_name) where chunks_by_set_name
        is keyed by chunk-set name -> chunk_id -> Chunk. Empty if
        chunks_save_path is None.
    """
    from rank_bm25 import BM25Okapi

    save_path = Path(save_path)
    manifest_path = save_path / "bm25_index_chunk_sets.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"BM25 index manifest not found at {manifest_path}.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    bm25_index_chunk_sets = []

    for entry in manifest:
        tokenized_chunks = [
            TokenizedChunk(
                chunk=Chunk(**tc["chunk"]),
                tokens=tc["tokens"],
            )
            for tc in entry["tokenized_chunks"]
        ]

        chunks_by_id: dict[int, Chunk] = {
            tc.chunk.chunk_id: tc.chunk for tc in tokenized_chunks
        }

        tokenized_documents = [tc.tokens for tc in tokenized_chunks]
        bm25 = BM25Okapi(tokenized_documents)

        bm25_index_chunk_sets.append(
            BM25IndexedChunkSet(
                name=entry["name"],
                index=bm25,
                tokenized_chunks=tokenized_chunks,
                chunks=chunks_by_id,
            )
        )

    chunks_by_set_name = _load_chunks_by_set_name(
        Path(chunks_save_path) if chunks_save_path is not None else None
    )
    return bm25_index_chunk_sets, chunks_by_set_name


def load_indexed_chunk_sets(
    save_path: str,
    chunks_save_path: str | None = None,
) -> tuple[list[IndexedChunkSet], dict[str, dict[int, Chunk]]]:
    """ Load indexed chunk sets from a snapshot.

    Args:
        save_path: directory holding the index manifest and .faiss files.
        chunks_save_path: directory holding chunk_sets.json. If provided,
            also load chunks and group them by set name so the caller can
            attach them to the returned IndexedChunkSets.

    Returns:
        (indexed_chunk_sets, chunks_by_set_name) where chunks_by_set_name
        is keyed by chunk-set name -> chunk_id -> Chunk. Empty if
        chunks_save_path is None.
    """
    save_path = Path(save_path)
    manifest_path = save_path / "indexed_chunk_sets.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Index manifest not found at {manifest_path}."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    indexed_chunk_sets = []

    for entry in manifest:
        index = faiss.read_index(str(save_path / entry["index"]))

        indexed_chunk_sets.append(
            IndexedChunkSet(
                name=entry["name"],
                index=index,
            )
        )

    chunks_by_set_name = _load_chunks_by_set_name(
        Path(chunks_save_path) if chunks_save_path is not None else None
    )
    return indexed_chunk_sets, chunks_by_set_name
