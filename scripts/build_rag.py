import mlflow
import argparse
import subprocess
from src.rag.core.utils import load_config
from rag.builder.base_builder import BaseBuilder

# Add logging
# Mlflow; calling it first-rag-experiement does not make sense. And should I be using Mlflow for every build?
# Snapshot has too many hardcoded save locations. These should be derived from configs. 
# What is going on with these the files config, files, dir?
# consider refactor related to snapshot.py
# take a look at dvc, this may have room for improvements
# base builders have configs.. should have a look at this. I think it is better if all configs are used at one place
# load_tokenized_chunk_sets, and load_encoded_chunk_sets can likely be combined into one, just need a key or extra parameter.

# after cleaning up, refactor configs.

# the strategy names; dense, hybrid, will be updated later. And thus the file names will be too.

PATHS_CONFIG = load_config("configs/paths_config.yaml")
BASE_CONFIG = load_config(PATHS_CONFIG["configs"]["base_config"])
BUILDING = PATHS_CONFIG["building"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        choices=["set_strategy", "chunk", "transform", "index", "seal_snapshot", "build"],
                        help="The RAG pipeline step to run.")
    return parser.parse_args()


# TODO: set_strategy should not live here.
def set_strategy() -> None:
    # get strategy from config
    strategy_name = BASE_CONFIG["rag"]["strategy"]

    BaseBuilder.set_strategy(
        strategy_name=strategy_name,
        metadata_path=BUILDING["metadata"]
    )


def chunk() -> None:
    builder = BaseBuilder.from_metadata(BUILDING["metadata"])
    builder.chunk()


def transform() -> None:
    builder = BaseBuilder.from_metadata(BUILDING["metadata"])
    builder.transform()

    
def index_embeddings()-> None:
    builder = BaseBuilder.from_metadata(BUILDING["metadata"])
    builder.index()
    

def finalise_build() -> None:
    builder = BaseBuilder.from_metadata(BUILDING["metadata"])
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

if __name__ == "__main__":
    args = parse_args()

    if args.run == "chunk":
        chunk()

    elif args.run == "transform":
        transform()

    elif args.run == "index":
        index_embeddings()

    elif args.run == "seal_snapshot":
        finalise_build()

    elif args.run == "build":
        build_rag()