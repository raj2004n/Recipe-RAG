import json
import logging
import os
import time
import mlflow
import argparse
import subprocess
from pathlib import Path

from src.rag.core.config import (
    BASE_CONFIG,
    BASE_CONFIG_FILE,
    PATHS_CONFIG_FILE,
    get_snapshot_paths,
    stage_config,
)
from src.rag.core.logging import configure_logging
from src.rag.core.mlflow_tracking import build_run_params
from src.rag.core.snapshot import lifecycle, registry
from src.rag.core.datasets.registry_dataset import register_dataset
from src.rag.core.datasets.resolver import LocalDatasetResolver
from src.rag.builder.base_builder import BaseBuilder


logger = logging.getLogger(__name__)


# Builds log straight to the local SQLite store by default, so no MLflow server
# needs to be running. Point RECIPE_RAG_MLFLOW_TRACKING_URI at an HTTP URI only
# when you want the run to go through a server (for example a live dashboard).
_MLFLOW_TRACKING_URI = os.environ.get("RECIPE_RAG_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
_MLFLOW_EXPERIMENT = "rag-builds"

# Where the build experiment keeps its artifacts. MLflow pins an experiment's
# artifact location when the experiment is created and has no supported way to
# change it afterwards, so the location is given here rather than left to the
# default: a store used locally gets a local root, and a build then writes its
# artifacts without an MLflow server needing to be reachable.
_MLFLOW_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "mlartifacts" / "rag-builds"

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
                             "Defaults to data.cleaned_recipes in configs/paths_config.yaml. "
                             "The file's checksum is captured at set_strategy so a mid-build "
                             "preprocessing swap cannot desync the snapshot.")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Override the auto-derived dataset key. Use to pin a key "
                             "across builds (e.g. 'recipes_v3').")
    return parser.parse_args()


def _registry_file() -> Path:
    return get_snapshot_paths().registry_file


def _sealed_name() -> str:
    """Return the versioned folder name of the newest sealed build."""
    name = registry.get_folder_for_target(_registry_file(), "latest_build")
    if not name:
        raise RuntimeError("No sealed build in the registry.")
    return name


def _dataset_key(metadata_path: Path) -> str:
    """Read the dataset key pinned into a build's metadata at set_strategy."""
    return json.loads(Path(metadata_path).read_text())["dataset_key"]


def _resolve_recipes_path() -> Path:
    """Resolve the dataset key pinned at set_strategy to a local path.

    Fail-closed: a missing key, missing file, or checksum drift raises. The
    lock that protected the file during the stage has been released by the time
    a later stage reads it, but the resolver still verifies the file is the same
    one the build registered.
    """
    key = _dataset_key(get_snapshot_paths().metadata)
    resolver = LocalDatasetResolver(
        registry_path=_registry_file(),
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(key)
    return Path(resolved.path)


def _prepare_experiment() -> None:
    """Create the build experiment with a local artifact root if it is missing.

    MLflow fixes an experiment's artifact location at creation and offers no
    supported API for changing it later, so an experiment created while a
    tracking server was in use carries an `mlflow-artifacts:` location that only
    that server resolves. A build logging to the local store then cannot write
    artifacts at all. Creating the experiment here gives a fresh store the local
    root; an existing experiment still pointing at a server is reported rather
    than repaired, because rewriting the tracking store is not something a build
    should do behind the operator's back.
    """
    experiment = mlflow.get_experiment_by_name(_MLFLOW_EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(_MLFLOW_EXPERIMENT, artifact_location=str(_MLFLOW_ARTIFACT_ROOT))
        return
    location = experiment.artifact_location
    if location.startswith("mlflow-artifacts:") and not _MLFLOW_TRACKING_URI.startswith("http"):
        logger.warning(
            "experiment %s stores artifacts at %s, which only an MLflow server resolves, so this "
            "run cannot write artifacts. Repoint it at %s before the next build.",
            _MLFLOW_EXPERIMENT, location, _MLFLOW_ARTIFACT_ROOT,
        )


def _snapshot_metrics(paths) -> dict[str, float]:
    """Counts and sizes describing what the build wrote, read back off disk.

    The numbers come from the sealed folder rather than from the builder, so a
    run's history describes the snapshot as it actually exists. Chunk counts are
    recorded per chunk set, and the file and byte totals cover the folder and its
    index directory.
    """
    metrics: dict[str, float] = {}

    for chunk_set in json.loads((paths.chunks / "chunk_sets.json").read_text()):
        metrics[f"counts.chunks.{chunk_set['name']}"] = len(chunk_set["chunks"])

    for label, manifest in (("dense", "indexed_chunk_sets.json"), ("bm25", "bm25_index_chunk_sets.json")):
        metrics[f"counts.indexes.{label}"] = len(json.loads((paths.index / manifest).read_text()))

    files = [path for path in paths.building.rglob("*") if path.is_file()]
    metrics["counts.files"] = len(files)
    metrics["bytes.snapshot"] = sum(path.stat().st_size for path in files)
    metrics["bytes.index"] = sum(
        path.stat().st_size for path in paths.index.rglob("*") if path.is_file()
    )
    return metrics


def set_strategy(recipes_path: Path | None = None, dataset_name: str | None = None) -> None:
    """Start a new build: register the dataset, open the staging tree, pin the key.

    The dataset is registered first (so the registry is the source of truth
    for what datasets exist), then `start_new_build` opens the staging tree and
    pins the key into the staged metadata. Capturing the key at this point - not
    at finalise - is what protects the build from a mid-flight preprocessing
    swap. No versioned folder is claimed here; the name is claimed when the
    build seals.
    """
    config = stage_config("set_strategy")
    recipes = (
        Path(recipes_path)
        if recipes_path is not None
        else Path(config.get(PATHS_CONFIG_FILE, "data.cleaned_recipes"))
    )
    dataset_key = register_dataset(
        registry_file=_registry_file(),
        path=recipes,
        name=dataset_name,
    )
    logger.info("registered dataset %s -> %s", dataset_key, recipes)

    paths = get_snapshot_paths()
    with lifecycle.staging_lock(paths.building):
        lifecycle.start_new_build(
            staging_dir=paths.building,
            strategy=config.get(BASE_CONFIG_FILE, "rag.strategy"),
            dataset_key=dataset_key,
        )
    logger.info("opened the staging tree at %s", paths.building)


def chunk() -> None:
    paths = get_snapshot_paths()
    recipes_path = _resolve_recipes_path()
    builder = BaseBuilder.from_paths(paths)
    with lifecycle.staging_lock(paths.building):
        with lifecycle.recipe_file_lock(recipes_path):
            builder.chunk(paths, recipes_path)


def transform() -> None:
    paths = get_snapshot_paths()
    builder = BaseBuilder.from_paths(paths)
    with lifecycle.staging_lock(paths.building):
        builder.transform(paths)


def index_stage() -> None:
    paths = get_snapshot_paths()
    builder = BaseBuilder.from_paths(paths)
    with lifecycle.staging_lock(paths.building):
        builder.index(paths)


def finalise_build() -> None:
    """Copy the staging tree into a freshly named sealed snapshot folder."""
    paths = get_snapshot_paths()
    builder = BaseBuilder.from_paths(paths)
    with lifecycle.staging_lock(paths.building):
        name = builder.seal_snapshot(
            paths=paths,
            registry_file=_registry_file(),
        )
    logger.info("sealed snapshot %s", name)


def build_rag(recipes_path: Path | None = None, dataset_name: str | None = None):
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    _prepare_experiment()
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
        logger.info("build finished in %.1fs", duration)

        # The build has succeeded by the time anything below runs, so a tracking
        # failure is a gap in the run's history rather than a failed build. One
        # guard around the whole of it keeps the run FINISHED and warns, instead
        # of leaving a sealed snapshot behind by a run marked FAILED.
        try:
            # Every stage has run by now, so the build this run produced is the
            # newest sealed one: finalise_build pointed `latest_build` at the
            # folder it copied the staging tree into.
            sealed = _sealed_name()
            paths = get_snapshot_paths(versioned_name=sealed)
            dataset_key = _dataset_key(paths.metadata)
            dataset = registry.get_dataset(_registry_file(), dataset_key) or {}

            # The dataset key and its checksum are read after the stages have
            # run, because set_strategy is what registers the dataset and pins
            # the key into the build's metadata.
            mlflow.log_param("snapshot_name", paths.building.name)
            mlflow.log_param("dataset_key", dataset_key)
            mlflow.log_param("dataset.checksum", dataset.get("checksum", ""))
            mlflow.log_metric("build.duration_sec", duration)
            mlflow.log_metrics(_snapshot_metrics(paths))

            # metadata.json names the build and the dataset it came from, and the
            # sealed configs are the copies the service itself reads. Those are
            # the artifacts worth carrying. The whole folder is not copied: 172 MB
            # of encoded vectors and index files already live in the snapshot and
            # in the DVC cache, and the artifact store is not a third home for
            # them.
            mlflow.log_artifact(str(paths.metadata))
            for config_file in sorted(paths.configs.glob("*.yaml")):
                mlflow.log_artifact(str(config_file))
        except Exception as e:
            logger.warning("build %s sealed but could not be recorded: %s", run_name, e)


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
