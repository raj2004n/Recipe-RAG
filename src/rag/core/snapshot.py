from dataclasses import dataclass, asdict
from typing import Callable
from ..retrieval.dense.encoder import EncodedChunk, EncodedChunkSet
from ..retrieval.dense.indexer import Indexer, IndexedChunkSet
from ..retrieval.chunker import Chunk, ChunkSet
import time
import uuid
import faiss
import json
import yaml
import os
from pathlib import Path
import numpy as np
import shutil

# TODO: Snapshot will be discarded
@dataclass
class Snapshot:
    # ids
    snapshot_id: str | None
    created_at: str | None

    # retriever state is per-set, so multiple chunk sets can be carried together
    chunk_sets: list[ChunkSet]
    indexed_chunk_sets: list[IndexedChunkSet]

    # configs
    base_config: dict[str, any]
    prompts_config: dict[str, any]

SAVE_PATH = os.path.join("snapshots/building")

def save_chunk_sets(chunk_sets: list[ChunkSet], save_path: str):
    save_path = os.path.join(save_path)
    os.makedirs(save_path, exist_ok=True)
    for chunk_set in chunk_sets:
        name = chunk_set.name
        with open(os.path.join(save_path, f"{name}_chunks.json"), "w") as f:
            json.dump([asdict(c) for c in chunk_set.chunks], f, indent=2)

def load_chunk_sets(path: str) -> list[ChunkSet]:
    path_obj = Path(path)
    # check if chunks directory exists
    if not path_obj.exists():
        raise FileNotFoundError(f"Chunks directory not found at {path}")

    chunks_sets = []

    # load all chunk files from directory
    for chunk_file in sorted(path_obj.glob("*.json")):
        name = chunk_file.stem.removesuffix("_chunks")
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = [Chunk(**c) for c in data]
            chunks_sets.append(
                ChunkSet(
                    name=name,
                    chunks=chunks
                )
            )
    return chunks_sets

def save_encoded_chunk_sets(encoded_chunks_sets: list[EncodedChunkSet], save_path: str):
    save_path = os.path.join(save_path)
    os.makedirs(save_path, exist_ok=True)
    for encoded_chunks_set in encoded_chunks_sets:
        name = encoded_chunks_set.name
        with open(os.path.join(save_path, f"{name}_encoded_chunks.json"), "w") as f:
            json.dump([asdict(c) for c in encoded_chunks_set.encoded_chunks], f, indent=2)

def load_encoded_chunk_sets(path: str) -> list[EncodedChunkSet]:
    path_obj = Path(path)
    # check if chunks directory exists
    if not path_obj.exists():
        raise FileNotFoundError(f"Chunks directory not found at {path}")

    encoded_chunks_sets = []

    # load all chunk files from directory
    for chunk_file in sorted(path_obj.glob("*.json")):
        name = chunk_file.stem.removesuffix("_encoded_chunks")
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            encoded_chunks = [EncodedChunk(**c) for c in data]
            encoded_chunks_sets.append(
                EncodedChunkSet(
                    name=name,
                    encoded_chunks=encoded_chunks
                )
            )
    return encoded_chunks_sets

def save_indexed_chunk_sets(indexed_chunk_sets: list[IndexedChunkSet], save_path: str):
    save_path = os.path.join(save_path)
    os.makedirs(save_path, exist_ok=True)
    for indexed_chunk_set in indexed_chunk_sets:
        name = indexed_chunk_set.name
        faiss.write_index(indexed_chunk_set.index, os.path.join(save_path, f"{name}_index.faiss"))
        np.save(
            os.path.join(save_path, f"{name}_index.mapping.npy"),
            indexed_chunk_set.id_mapping,
            allow_pickle=True
        )

def load_indexed_chunk_sets(path: str) -> list[IndexedChunkSet]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Index directory not found at {path}.")

    indexed_chunk_sets = []

    for index_file in sorted(path_obj.glob("*_index.faiss")):
        name = index_file.stem.removesuffix("_index")
        index = faiss.read_index(str(index_file))

        mapping = np.load(
            path_obj / f"{name}_index.mapping.npy",
            allow_pickle=True,
        ).item()

        indexed_chunk_sets.append(
            IndexedChunkSet(
                name=name,
                index=index,
                id_mapping=mapping,
            )
        )

    return indexed_chunk_sets

def save_config(save_path: str, config_path: str):
    save_dir = Path(save_path)
    src_dir = Path(config_path)

    save_dir.mkdir(parents=True, exist_ok=True)

    dst = save_dir / src_dir.name

    shutil.copy2(src_dir, dst)

def finalise_snapshot(snapshot_id: str | None = None, created_at: str | None = None):
    building = Path("snapshots/building")
    latest = Path("snapshots/latest")

    if not building.exists():
        raise FileNotFoundError(f"Building directory not found at {building}")

    if building.exists() and latest.exists():
        raise FileExistsError(f"Latest snapshot already exists at {latest}")

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

    # add metadata
    metadata_path = os.path.join("snapshots/building", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "snapshot_id": snapshot_id,
            "created_at": created_at,
        }, f, indent=2)

    # rename latest (old) snapshot to its id
    rename_last_snapshot()

    latest = Path("snapshots/latest")
    if latest.exists():
        shutil.rmtree(latest)

    building.rename(latest)

def load_snapshot(path: str) -> Snapshot:
    """Load a snapshot produced by the per-set build pipeline.

    Layout (per-set):
        <path>/configs/base_config.yaml
        <path>/configs/prompts_config.yaml
        <path>/configs/paths_config.yaml
        <path>/chunks/<name>_chunks.json
        <path>/encoded_chunks/<name>_encoded_chunks.json
        <path>/index/<name>_index.faiss
        <path>/index/<name>_index.mapping.npy
        <path>/metadata.json
    """
    path_obj = Path(path)

    # load configs
    with open(path_obj / "configs" / "base_config.yaml") as f:
        base_config = yaml.safe_load(f)
    with open(path_obj / "configs" / "prompts_config.yaml") as f:
        prompts_config = yaml.safe_load(f)

    # load chunk sets
    chunk_sets = load_chunk_sets(path_obj / "chunks")

    # load indexed chunk sets
    indexed_chunk_sets = load_indexed_chunk_sets(path_obj / "index")

    # load metadata
    with open(path_obj / "metadata.json") as f:
        metadata = json.load(f)

    return Snapshot(
        snapshot_id=metadata["snapshot_id"],
        created_at=metadata["created_at"],
        chunk_sets=chunk_sets,
        indexed_chunk_sets=indexed_chunk_sets,
        base_config=base_config,
        prompts_config=prompts_config,
    )

def rename_last_snapshot():
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
        raise FileExistsError(f"Snapshot with id {snapshot_id} already exists")

    os.rename(latest_snapshot_path, new_snapshot_path)

def snapshot_retriever_indexer(
    snapshot: Snapshot,
    accepts: Callable[[IndexedChunkSet], bool] = lambda _: True,
) -> Indexer:
    """Build an Indexer whose indexed_sets match the snapshot's stored sets
    and whose id_mapping matches each set's chunk ids.

    The same accepts predicate used at build time must be passed in here so
    the runtime indexer only contains the sets the retriever cares about.
    """
    base = snapshot.base_config
    indexer = Indexer(
        index_type=base["indexer"]["type"],
        dimension=base["indexer"]["dimension"],
        accepts=accepts,
    )
    # Reuse the per-set indexes from the snapshot, but filtered through accepts
    # so the retriever never wastes cycles on sets it wouldn't look up against.
    indexer.indexed_sets = [
        s for s in snapshot.indexed_chunk_sets if accepts(s)
    ]
    return indexer
