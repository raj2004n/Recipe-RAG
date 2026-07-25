import subprocess
import mlflow
import argparse
from pathlib import Path
from rag.retrieval.dense.encoder import Encoder
from rag.retrieval.dense.indexer import Indexer
from src.rag.retrieval.chunker import chunk_recipes
from src.rag.core.utils import load_config
from src.rag.core.snapshot import save_chunk_sets, save_encoded_chunk_sets, load_chunk_sets, load_encoded_chunk_sets, save_indexed_chunk_sets, finalise_snapshot

#TODO: Add logging

MODEL_CONFIG = load_config("configs/base_config.yaml")
PROMPTS_CONFIG = load_config("configs/prompts_config.yaml")
PATHS_CONFIG = load_config("configs/paths_config.yaml")

FILES = PATHS_CONFIG["files"]
DIRECTORIES = PATHS_CONFIG["directories"]

ENCODER_CONFIG = MODEL_CONFIG["encoder"]
INDEXER_CONFIG = MODEL_CONFIG["indexer"]
RETRIEVER_CONFIG = MODEL_CONFIG["retriever"]
GENERATOR_CONFIG = MODEL_CONFIG["generator"]

PROMPT_TEMPLATES = PROMPTS_CONFIG["template"]
PROMPT_SETTINGS = PROMPTS_CONFIG["settings"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        choices=["chunk", "encode", "index", "finalise_snapshot", "build"],
                        help="The RAG pipeline step to run.")
    return parser.parse_args()

def chunk() -> None:
    chunk_sets = chunk_recipes(MODEL_CONFIG["rag"]["strategy"], FILES["cleaned_recipes"])
    save_chunk_sets(
        chunk_sets=chunk_sets, 
        save_path=PATHS_CONFIG["building"]["chunks"]
    )

def encode_chunk() -> None:
    chunks_path = PATHS_CONFIG["building"]["chunks"]

    chunks_sets = load_chunk_sets(chunks_path)

    encoder = Encoder(
        model_id=ENCODER_CONFIG["model_id"],
        batch_size=ENCODER_CONFIG["batch_size"],
        normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"]
    )

    encoded_chunks_sets = encoder.encode_chunks(chunks_sets)

    save_encoded_chunk_sets(
        encoded_chunks_sets=encoded_chunks_sets,
        save_path=PATHS_CONFIG["building"]["encoded_chunks"]
    )

def index_embeddings() -> None:
    encoded_chunks_sets = load_encoded_chunk_sets(PATHS_CONFIG["building"]["encoded_chunks"])

    indexer = Indexer(
        index_type=INDEXER_CONFIG["type"],
        dimension=INDEXER_CONFIG["dimension"]
    )

    indexed_chunks_sets = indexer.index_embeddings(encoded_chunks_sets)

    save_indexed_chunk_sets(
        indexed_chunk_sets=indexed_chunks_sets,
        save_path=PATHS_CONFIG["building"]["index"]
    )

def complete_snapshot():
    return finalise_snapshot(
        snapshot_id=None,
        created_at=None
    )

def build_rag():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("first-rag-experiment")

    subprocess.run(["dvc", "repro", "chunk"])
    subprocess.run(["dvc", "repro", "encode"])
    subprocess.run(["dvc", "repro", "index"])
    subprocess.run(["dvc", "repro", "finalise_snapshot"])

    mlflow.log_artifact("snapshots/latest")

if __name__ == "__main__":
    args = parse_args()
    if args.run == "chunk":
        chunk()
    elif args.run == "encode":
        encode_chunk()
    elif args.run == "index":
        index_embeddings()
    elif args.run == "finalise_snapshot":
        complete_snapshot()
    elif args.run == "build":
        build_rag()