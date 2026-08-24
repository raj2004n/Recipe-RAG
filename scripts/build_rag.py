import json
import logging
import time
import mlflow
import argparse
import subprocess
from pathlib import Path

from src.rag.core.config import BASE_CONFIG, DATA_PATHS, get_snapshot_paths
from src.rag.core.logging import configure_logging
from src.rag.core.mlflow_tracking import build_run_params
from src.rag.core.snapshot import lifecycle, registry
from src.rag.core.datasets.registry_dataset import register_dataset
from src.rag.core.datasets.resolver import LocalDatasetResolver
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
    parser.add_argument("--recipes-path", type=str, default=None,
                        help="Path to a recipes dataset to register for this build. "
                             "Defaults to DATA_PATHS['cleaned_recipes']. The file's "
                             "checksum is captured at set_strategy so a mid-build "
                             "preprocessing swap cannot desync the snapshot.")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Override the auto-derived dataset key. Use to pin a key "
                             "across builds (e.g. 'recipes_v3').")
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


def _active_dataset_key() -> str:
    """Read the dataset key captured at set_strategy from the active build."""
    name = _current_versioned_name()
    metadata_path = _SNAPSHOTS_DIR / name / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    return metadata["dataset_key"]


def _resolve_recipes_path() -> Path:
    """Resolve the dataset key captured at set_strategy to a local path.

    Fail-closed: a missing key, missing file, or checksum drift raises. The
    lock that protected the file during the build has been released by now
    (finalise_snapshot closes it), but the resolver still verifies the file
    is the same one we registered.
    """
    key = _active_dataset_key()
    resolver = LocalDatasetResolver(
        registry_path=_registry_file(),
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(key)
    return Path(resolved.path)


def set_strategy(recipes_path: Path | None = None, dataset_name: str | None = None) -> None:
    """Start a new build: register the dataset, claim a versioned folder, write metadata.

    The dataset is registered first (so the registry is the source of truth
    for what datasets exist), then `start_new_build` pins the key into the
    snapshot metadata. Capturing the key at this point — not at finalise —
    is what protects the build from a mid-flight preprocessing swap.
    """
    recipes = Path(recipes_path) if recipes_path is not None else Path(DATA_PATHS["cleaned_recipes"])
    dataset_key = register_dataset(
        registry_file=_registry_file(),
        path=recipes,
        name=dataset_name,
    )
    logger.info("registered dataset %s -> %s", dataset_key, recipes)

    name = lifecycle.start_new_build(
        snapshots_dir=_SNAPSHOTS_DIR,
        registry_file=_registry_file(),
        strategy=BASE_CONFIG["rag"]["strategy"],
        dataset_key=dataset_key,
    )
    logger.info("started new build %s", name)


def chunk() -> None:
    paths = get_snapshot_paths(versioned_name=_current_versioned_name())
    recipes_path = _resolve_recipes_path()
    builder = BaseBuilder.from_paths(paths)
    with lifecycle.recipe_file_lock(recipes_path):
        builder.chunk(paths, recipes_path)


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
    # is not installed or the working tree is not a DVC repo. Pass an
    # absolute path so dvc does not interpret a relative path under cwd.
    try:
        subprocess.run(
            ["dvc", "add", str(paths.building.resolve())],
            check=True,
        )
    except FileNotFoundError:
        logger.warning("dvc not on PATH; skipping dvc add for %s", paths.building)
    except subprocess.CalledProcessError as e:
        logger.warning("dvc add failed for %s: %s", paths.building, e)


def build_rag(recipes_path: Path | None = None, dataset_name: str | None = None):
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


def test_build_rag(recipes_path: Path | None = None, dataset_name: str | None = None):
    set_strategy(recipes_path=recipes_path, dataset_name=dataset_name)
    chunk()
    transform()
    index_stage()
    finalise_build()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    if args.run == "set_strategy":
        set_strategy(recipes_path=args.recipes_path, dataset_name=args.dataset_name)
    elif args.run == "chunk":
        chunk()
    elif args.run == "transform":
        transform()
    elif args.run == "index":
        index_stage()
    elif args.run == "finalise_build":
        finalise_build()
    elif args.run == "build":
        build_rag(recipes_path=args.recipes_path, dataset_name=args.dataset_name)
    elif args.run == "test_build":
        test_build_rag(recipes_path=args.recipes_path, dataset_name=args.dataset_name)
