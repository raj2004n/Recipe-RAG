""" 
Contains snapshot save logic. 

Usage:
The retriever is built in modular steps; chunks, encode chunks, index chunks, ...
At the end of each step, the respsective save mehtod is called.

Therefore, a 'one-does-it-all' function, e.g. save_snapshot, does not exist.

But for inference, the retriever is loaded in one step using load_snapshot.
"""


from pathlib import Path
from dataclasses import dataclass, asdict

from ..retrieval.chunker import Chunk, ChunkSet
from ..retrieval.dense.encoder import EncodedChunk, EncodedChunkSet
from ..retrieval.dense.indexer import IndexedChunkSet
from ..retrieval.sparse.bm25indexer import BM25IndexedChunkSet
from ..retrieval.sparse.tokenizer import TokenizedChunk, TokenizedChunkSet

import os
import time
import uuid
import faiss
import json
import yaml
import shutil


@dataclass
class Snapshot:
    snapshot_id: str | None
    created_at: str | None
    retrieval_strategy: str
    indexed_chunk_sets: list[IndexedChunkSet] # carries chunks in IndexedChunkSet.chunks
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet] # carries chunks in BM25IndexedChunkSet.chunks
    base_config: dict[str, any] # configs for initialising retriever components for inference, in pipeline
    prompts_config: dict[str, any]


def save_chunk_sets(chunk_sets: list[ChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "chunk_sets.json"), "w") as f:
        json.dump([asdict(cs) for cs in chunk_sets], f, indent=2)


def load_chunk_sets(save_path: str) -> list[ChunkSet]:
    with open(os.path.join(save_path, "chunk_sets.json")) as f:
        data = json.load(f)

    return [
        ChunkSet(
            name=cs["name"],
            chunks=[Chunk(**chunk) for chunk in cs["chunks"]],
        )
        for cs in data
    ]


def save_encoded_chunk_sets(encoded_chunk_sets: list[EncodedChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "encoded_chunk_sets.json"), "w") as f:
        json.dump([asdict(cs) for cs in encoded_chunk_sets], f, indent=2)


def save_tokenized_chunk_sets(tokenized_chunk_sets: list[TokenizedChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "tokenized_chunk_sets.json"), "w",) as f:
        json.dump([asdict(cs) for cs in tokenized_chunk_sets], f, indent=2)


def load_encoded_chunk_sets(save_path: str) -> list[EncodedChunkSet]:
    with open(os.path.join(save_path, "encoded_chunk_sets.json"), "r") as f:
        data = json.load(f)

    return [
        EncodedChunkSet(
            name=chunk_set["name"],
            encoded_chunks=[
                EncodedChunk(**chunk)
                for chunk in chunk_set["encoded_chunks"]
            ],
        )
        for chunk_set in data
    ]


def load_tokenized_chunk_sets(save_path: str) -> list[TokenizedChunkSet]:
    with open(os.path.join(save_path, "tokenized_chunk_sets.json"), "r") as f:
        data = json.load(f)

    return [
        TokenizedChunkSet(
            name=chunk_set["name"],
            tokenized_chunks=[
                TokenizedChunk(
                    chunk=Chunk(**tokenized_chunk["chunk"]),
                    tokens=tokenized_chunk["tokens"],
                )
                for tokenized_chunk in chunk_set["tokenized_chunks"]
            ],
        )
        for chunk_set in data
    ]


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


def load_bm25_index_chunk_sets(
    save_path: str,
    chunks_save_path: str | None = None,
) -> tuple[list[BM25IndexedChunkSet], dict[str, dict[int, "Chunk"]]]:
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

    chunks_by_set_name: dict[str, dict[int, Chunk]] = {}
    if chunks_save_path is not None:
        chunk_sets = load_chunk_sets(chunks_save_path)
        for cs in chunk_sets:
            chunks_by_set_name[cs.name] = {c.chunk_id: c for c in cs.chunks}

    return bm25_index_chunk_sets, chunks_by_set_name


def load_indexed_chunk_sets(
    save_path: str,
    chunks_save_path: str | None = None,
) -> tuple[list[IndexedChunkSet], dict[str, dict[int, "Chunk"]]]:
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
        index = faiss.read_index(
            str(save_path / entry["index"])
        )

        indexed_chunk_sets.append(
            IndexedChunkSet(
                name=entry["name"],
                index=index,
            )
        )

    chunks_by_set_name: dict[str, dict[int, Chunk]] = {}
    if chunks_save_path is not None:
        chunk_sets = load_chunk_sets(chunks_save_path)
        for cs in chunk_sets:
            chunks_by_set_name[cs.name] = {c.chunk_id: c for c in cs.chunks}

    return indexed_chunk_sets, chunks_by_set_name


def save_config(save_path: str, config_path: str):
    save_dir = Path(save_path)
    src_dir = Path(config_path)

    save_dir.mkdir(parents=True, exist_ok=True)

    dst = save_dir / src_dir.name

    shutil.copy2(src_dir, dst)


def rename_latest_snapshot():
    # change the name of the latest snapshot to its id
    latest_snapshot_path = os.path.join("snapshots", "latest")
    if not os.path.exists(latest_snapshot_path):
        # no latest snapshot
        return

    metadata_path = os.path.join(latest_snapshot_path, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata file found at {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    snapshot_id = metadata["snapshot_id"]
    new_snapshot_path = os.path.join("snapshots", snapshot_id)

    if os.path.exists(new_snapshot_path):
        # if an archive with this id already exists then add a timestamp
        # suffix rather than overwriting or aborting the build
        suffix = time.strftime("%Y%m%d%H%M%S")
        new_snapshot_path = os.path.join("snapshots", f"{snapshot_id}_{suffix}")
        print(
            f"Warning: snapshot with id {snapshot_id} already exists; "
            f"archiving latest to {os.path.basename(new_snapshot_path)}"
        )

    os.rename(latest_snapshot_path, new_snapshot_path)


def finalise_snapshot(snapshot_id: str | None = None, created_at: str | None = None):
    """ Used at the last step of building retriever to finalise the 'building' snapshot to 'latest' snapshot.
    Saves config, adds metadata containing snapshot id and date. Renames the building snapshot to latest."""
    building = Path("snapshots/building")
    latest = Path("snapshots/latest")

    if not building.exists():
        raise FileNotFoundError(f"Building directory not found at {building}")

    rename_latest_snapshot()

    configs_save_path = "snapshots/building/configs"
    save_config(
        save_path=configs_save_path,
        config_path="configs/base_config.yaml"
    )
    save_config(
        save_path=configs_save_path,
        config_path="configs/prompts_config.yaml"
    )
    save_config(
        save_path=configs_save_path,
        config_path="configs/paths_config.yaml"
    )

    # get id and timestamp if not provided
    snapshot_id = snapshot_id or str(uuid.uuid4())
    created_at = created_at or time.strftime("%Y-%m-%d %H:%M:%S")

    metadata_path = os.path.join("temp/building", "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata.update({
        "snapshot_id": snapshot_id,
        "created_at": created_at,
    })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    building.rename(latest)


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
    indexed_chunk_sets: list[IndexedChunkSet] = []
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet] = []

    # dense strategies include "dense" or "hybrid"
    if "dense" in strategy or strategy == "hybrid":
        indexed_chunk_sets, chunks_by_set_name = load_indexed_chunk_sets(
            path_obj / "index",
            chunks_save_path=path_obj / "chunks",
        )
        for indexed in indexed_chunk_sets:
            indexed.chunks = chunks_by_set_name.get(indexed.name, {})

    # sparse/BM25 strategies include "sparse", "bm25", or "hybrid"
    if strategy == "hybrid":
        bm25_indexed_chunk_sets, bm25_chunks_by_set_name = load_bm25_index_chunk_sets(
            path_obj / "index",
            chunks_save_path=path_obj / "chunks",
        )
        for indexed in bm25_indexed_chunk_sets:
            indexed.chunks = bm25_chunks_by_set_name.get(indexed.name, {})

    return Snapshot(
        snapshot_id=metadata["snapshot_id"],
        created_at=metadata["created_at"],
        retrieval_strategy=strategy,
        indexed_chunk_sets=indexed_chunk_sets,
        bm25_indexed_chunk_sets=bm25_indexed_chunk_sets,
        base_config=base_config,
        prompts_config=prompts_config,
    )