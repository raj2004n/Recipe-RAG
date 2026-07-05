import yaml
import json
import time
from pathlib import Path
from src.retrieval.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.retrieval.encoder import Encoder

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

def load_retriever(config: dict) -> tuple[Retriever, float, float, float]:
    
    start = time.time()
    chunks = load_chunks()
    end = time.time()
    chunks_time = end - start

    encoder = Encoder(config) # uses pretrained model, hence no load
    
    start2 = time.time()
    indexer = load_indexer()
    end2 = time.time()
    index_time = end - start

    retriever = Retriever(
        encoder=encoder,
        indexer=indexer,
        chunks=chunks,    
        config=config
    )
    end3 = time.time()
    retriever_time = end3 - end2

    return retriever, chunks_time, index_time, retriever_time

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