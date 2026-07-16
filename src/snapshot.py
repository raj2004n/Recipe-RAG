from dataclasses import dataclass, asdict
from .retrieval.indexer import Indexer
from .retrieval.chunker import Chunk
import time
import uuid
import faiss
import json 
import yaml
import os

@dataclass
class Snapshot:
    # ids
    snapshot_id: str | None
    created_at: str | None

    # retriever
    index: Indexer
    chunks: list[Chunk]

    # configs
    base_config: dict[str, any]
    prompts_config: dict[str, any]


def save_snapshot(snapshot: Snapshot, path: str):
    # rename latest snapshot to its id
    rename_last_snapshot()

    # get id and timestamp if not provided
    snapshot_id = snapshot.snapshot_id or str(uuid.uuid4())
    created_at = snapshot.created_at or time.strftime("%Y-%m-%d %H:%M:%S")

    # save snapshot to latest
    path = os.path.join(path, "latest")
    os.makedirs(path, exist_ok=True)

    # save chunks
    with open(os.path.join(path, "chunks.json"), "w") as f:
        json.dump([asdict(c) for c in snapshot.chunks], f, indent=2)
    
    # save index
    snapshot.index.save(os.path.join(path, "index.faiss"))
    
    # save configs
    with open(os.path.join(path, "base_config.yaml"), "w") as f:
        yaml.dump(snapshot.base_config, f, sort_keys=False)
    
    with open(os.path.join(path, "prompts_config.yaml"), "w") as f:
        yaml.dump(snapshot.prompts_config, f, sort_keys=False)

    # save metadata
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump({
            "snapshot_id": snapshot_id,
            "created_at": created_at,
        }, f, indent=2)

def load_snapshot(path: str) -> Snapshot:
    # load chunks
    with open(os.path.join(path, "chunks.json")) as f:
        chunks = json.load(f)
    
    # load index
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    
    # load configs
    with open(os.path.join(path, "base_config.yaml")) as f:
        base_config = yaml.safe_load(f)
    
    with open(os.path.join(path, "prompts_config.yaml")) as f:
        prompts_config = yaml.safe_load(f)
    
    # load metadata
    with open(os.path.join(path, "metadata.json")) as f:
        metadata = json.load(f)

    return Snapshot(
        snapshot_id=metadata["snapshot_id"],
        created_at=metadata["created_at"],
        index=index,
        chunks=chunks,
        base_config=base_config,
        prompts_config=prompts_config
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
