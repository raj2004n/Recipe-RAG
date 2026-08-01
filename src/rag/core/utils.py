import yaml
import json
from pathlib import Path
from rag.retrieval.dense.indexer import Indexer
from rag.retrieval.dense.retriever import Retriever
from rag.retrieval.dense.encoder import Encoder
from src.rag.core.snapshot import Snapshot, snapshot_retriever_indexer

def save_config(directory: Path, config: dict):
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def load_config(config_file: str = "configs/base_config.yaml") -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def dense_accepts_for(snapshot: Snapshot):
    """Build the dense-set accepts predicate from the snapshot's config."""
    names = set(snapshot.base_config.get("retriever", {}).get("dense_sets", ["dense"]))
    return lambda s: s.name in names

def load_retriever(snapshot: Snapshot) -> Retriever:
    accepts = dense_accepts_for(snapshot)
    encoder = Encoder(
        snapshot.base_config["encoder"]["model_id"],
        snapshot.base_config["encoder"]["batch_size"],
        snapshot.base_config["encoder"]["normalize_embeddings"],
        accepts=accepts,
    )
    indexer = snapshot_retriever_indexer(snapshot, accepts=accepts)
    retriever = Retriever(
        snapshot.chunk_sets,
        encoder,
        indexer,
        snapshot.base_config["retriever"]["top_k"],
        snapshot.base_config["retriever"]["score_threshold"],
        accepts=accepts,
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