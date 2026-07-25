import yaml
import json
from pathlib import Path
from rag.retrieval.dense.indexer import Indexer
from rag.retrieval.dense.retriever import Retriever
from rag.retrieval.dense.encoder import Encoder
from src.rag.core.snapshot import Snapshot

def save_config(directory: Path, config: dict):
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def load_config(config_file: str = "configs/base_config.yaml") -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

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