import logging
import time
import mlflow
import argparse
import subprocess
from pathlib import Path

from src.rag.core.config import BASE_CONFIG, get_snapshot_paths
from src.rag.core.logging import configure_logging
from src.rag.core.mlflow_tracking import build_run_params
from src.rag.core.snapshot import registry
from src.rag.builder.base_builder import BaseBuilder


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
                        choices=["set_strategy", "chunk", "transform", "index", "finalise_build", "build", "test_build"],
                        help="The RAG pipeline step to run.")
    return parser.parse_args()


_SNAPSHOTS_DIR = Path("snapshots")


def _registry_file() -> Path:
    return _SNAPSHOTS_DIR / "registry.json"


def _current_versioned_name() -> str:
    """Return the versioned folder name registered as the active build.

    Raises RuntimeError if set_strategy has not been run.
    """
    name = registry.get_folder_for_target(_registry_file(), "current")
    if not name:
        raise RuntimeError(
            "No active build in the registry. Run "
            "'python -m scripts.build_rag --run set_strategy' first."
        )
    return name


def set_strategy() -> None:
    """Start a new build: claim a versioned folder, write metadata, mark current.
    """
    name = BaseBuilder.set_strategy_from_config(
        BASE_CONFIG,
        snapshots_dir=_SNAPSHOTS_DIR,
        registry_file=_registry_file(),
    )
    logger.info("started new build %s", name)


def chunk() -> None:
    paths = get_snapshot_paths(versioned_name=_current_versioned_name())
    builder = BaseBuilder.from_paths(paths)
    builder.chunk(paths)


def transform() -> None:
    paths = get_snapshot_paths(versioned_name=_current_versioned_name())
    builder = BaseBuilder.from_paths(paths)
    builder.transform(paths)


def index_stage() -> None:
    paths = get_snapshot_paths(versioned_name=_current_versioned_name())
    builder = BaseBuilder.from_paths(paths)
    builder.index(paths)


def finalise_build() -> None:
    paths = get_snapshot_paths(versioned_name=_current_versioned_name())
    builder = BaseBuilder.from_paths(paths)
    builder.seal_snapshot(
        paths=paths,
        registry_file=_registry_file(),
        versioned_name=paths.building.name,
    )
    # Register the completed snapshot with DVC so its contents are
    # content-hashed and tracked for reproducibility. Skip silently if dvc
    # is not installed or the working tree is not a DVC repo.
    try:
        subprocess.run(
            ["dvc", "add", str(paths.building)],
            check=True,
            cwd=str(_SNAPSHOTS_DIR),
        )
    except FileNotFoundError:
        logger.warning("dvc not on PATH; skipping dvc add for %s", paths.building)
    except subprocess.CalledProcessError as e:
        logger.warning("dvc add failed for %s: %s", paths.building, e)


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

        paths = get_snapshot_paths(versioned_name=_current_versioned_name())
        mlflow.log_param("snapshot_name", paths.building.name)
        mlflow.log_artifact(str(paths.building / "metadata.json"))
        try:
            mlflow.log_artifact(str(paths.building), artifact_path="snapshot")
        except Exception as e:
            logger.warning("failed to log snapshot artifact: %s", e)
        mlflow.log_artifact("configs/base_config.yaml")
        mlflow.log_artifact("configs/paths_config.yaml")
        mlflow.log_artifact("configs/prompts_config.yaml")


def test_build_rag():
    set_strategy()
    chunk()
    transform()
    index_stage()
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
        index_stage()
    elif args.run == "finalise_build":
        finalise_build()
    elif args.run == "build":
        build_rag()
    elif args.run == "test_build":
        test_build_rag()
