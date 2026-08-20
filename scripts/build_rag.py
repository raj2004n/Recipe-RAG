import time
import logging
import mlflow
import argparse
import subprocess
from src.rag.core.config import BASE_CONFIG, BUILDING_PATHS, SNAPSHOT_PATHS
from src.rag.core.logging import configure_logging
from src.rag.core.mlflow_tracking import build_run_params
from src.rag.builder.base_builder import BaseBuilder

# take a look at dvc, this may have room for improvements

logger = logging.getLogger(__name__)


_MLFLOW_TRACKING_URI = "http://localhost:5000"
_MLFLOW_EXPERIMENT = "rag-builds"

_DVC_STAGES = [
    "set_strategy",
    "chunk",
    "transform",
    "index",
    "finalise_build",
]


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
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    strategy = BASE_CONFIG["rag"]["strategy"]
    run_name = f"build-{strategy}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("starting build run %s", run_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(build_run_params())

        t0 = time.time()
        for stage in _DVC_STAGES:
            logger.info("running dvc stage %s", stage)
            subprocess.run(["dvc", "repro", stage], check=True)
        duration = time.time() - t0
        mlflow.log_metric("build.duration_sec", duration)
        logger.info("build finished in %.1fs", duration)

        mlflow.log_artifact(str(SNAPSHOT_PATHS.latest), artifact_path="snapshot")
        mlflow.log_artifact("configs/base_config.yaml")
        mlflow.log_artifact("configs/paths_config.yaml")
        mlflow.log_artifact("configs/prompts_config.yaml")


def test_build_rag():
    set_strategy()
    chunk()
    transform()
    index()
    finalise_build()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    if args.run == "set_strategy":
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
