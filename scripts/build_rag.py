import mlflow
import argparse
import subprocess
from src.rag.core.config import BASE_CONFIG, BUILDING_PATHS
from src.rag.builder.base_builder import BaseBuilder

# Add logging
# Mlflow; calling it first-rag-experiement does not make sense. And should I be using Mlflow for every build?
# take a look at dvc, this may have room for improvements
# load_tokenized_chunk_sets, and load_encoded_chunk_sets can likely be combined into one, just need a key or extra parameter.
# the strategy names; dense, hybrid, will be updated later. And thus the file names will be too.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        choices=["set_strategy", "chunk", "transform", "index", "finalise_build", "build", "test_build"], # test_build does not use dvc or mlflow
                        help="The RAG pipeline step to run.")
    return parser.parse_args()


def set_strategy() -> None:
    BaseBuilder.set_strategy_from_config(BASE_CONFIG, BUILDING_PATHS["metadata"])


def chunk() -> None:
    builder = BaseBuilder.from_metadata(BUILDING_PATHS["metadata"])
    builder.chunk()


def transform() -> None:
    builder = BaseBuilder.from_metadata(BUILDING_PATHS["metadata"])
    builder.transform()


def index()-> None:
    builder = BaseBuilder.from_metadata(BUILDING_PATHS["metadata"])
    builder.index()


def finalise_build() -> None:
    builder = BaseBuilder.from_metadata(BUILDING_PATHS["metadata"])
    builder.seal_snapshot()


def build_rag():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("first-rag-experiment")

    subprocess.run(["dvc", "repro", "set_strategy"])
    subprocess.run(["dvc", "repro", "chunk"])
    subprocess.run(["dvc", "repro", "transform"])
    subprocess.run(["dvc", "repro", "index"])
    subprocess.run(["dvc", "repro", "finalise_build"])

    mlflow.log_artifact("snapshots/latest")


def test_build_rag():
    set_strategy()
    chunk()
    transform()
    index()
    finalise_build()


if __name__ == "__main__":
    args = parse_args()
    if args.run == "set_straetgy":
        set_strategy()

    elif args.run == "chunk":
        chunk()

    elif args.run == "transform":
        transform()

    elif args.run == "index":
        index()

    elif args.run == "finalise_build":
        finalise_build()

    elif args.run == "build":
        build_rag()

    elif args.run == "test_build":
        test_build_rag()