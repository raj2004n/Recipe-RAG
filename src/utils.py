import os
import yaml
import json
import time
import uuid
import faiss
from pathlib import Path
from dataclasses import asdict
from src.retrieval.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.retrieval.encoder import Encoder
from src.data_classes import Snapshot

def save_config(directory: Path, config: dict):
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def load_config(config_file: str = "configs/base_config.yaml") -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_chunks(path: str = "models/retriever/chunks.json") -> list[dict]:
    with open(path) as f:
        chunks = json.load(f)
    return chunks

def load_indexer(path: str = "models/retriever/index.faiss") -> Indexer:
    indexer = Indexer(1) # placeholder, overwrite with load
    indexer.load(path)
    return indexer

def load_retriever(snapshot: Snapshot) -> Retriever:
    encoder = Encoder(
        snapshot.base_config["encoder"]["model"],
        snapshot.base_config["encoder"]["batch_size"],
        snapshot.base_config["encoder"]["normalize_embeddings"]
    )
    indexer = Indexer.load(
        snapshot.base_config["paths"]["files"]["indexer"],
        snapshot.base_config["indexer"]["type"],
        snapshot.base_config["indexer"]["dimension"]
    )
    retriever = Retriever(
        snapshot.chunks,
        encoder,
        indexer,
        snapshot.base_config["retriever"]["top_k"],
        snapshot.base_config["retriever"]["score_threshold"]
    )
    return retriever

def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    return dataset

def flatten_config_dict(d, parent_key="", sep="."):
    items = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_config_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items

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
