"""Evaluate retrieval for a snapshot against the eval datasets, tracked in MLflow.

Loads a sealed snapshot, builds the pipeline for its strategy, and scores
retrieval against the golden and archived eval slices. The two slices are
evaluated and reported separately, never blended. Each run logs the snapshot
identity, the retrieval config, the per-slice and per-query-type metrics, and
the eval files plus per-query results as artifacts.
"""

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import mlflow

from src.rag.core.config import DATA_PATHS
from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import loader, registry
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.core.datasets.resolver import LocalDatasetResolver
from src.rag.evaluation.retriever_evaluator import RetrievalEvaluator


logger = logging.getLogger(__name__)

_MLFLOW_TRACKING_URI = "http://localhost:5000"
_MLFLOW_EXPERIMENT = "rag-retrieval-eval"

# Slice name -> eval file. Golden and archived are reported separately and
# never blended.
_EVAL_SLICES = {
    "golden": DATA_PATHS["eval_golden_prompts"],
    "archived": DATA_PATHS["eval_archived_prompts"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=str, default="latest",
        help="Snapshot target ('latest' or 'current') or folder name to evaluate.",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10],
        help="k values to score. Defaults to 1 2 3 5 10.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of rows evaluated per slice (for smoke runs).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Skip MLflow logging; print the results instead.",
    )
    return parser.parse_args()


def _resolve_snapshot_folder(target: str) -> str:
    """Resolve a target name or folder name to a snapshot folder name."""
    if target in ("latest", "current"):
        paths = SnapshotPaths.from_paths_config({})
        folder = registry.get_folder_for_target(paths.registry_file, target)
        if not folder:
            raise FileNotFoundError(
                f"No snapshot folder mapped to target {target!r} in registry"
            )
        return folder
    return target


def _load_snapshot(folder: str):
    paths = SnapshotPaths.from_paths_config({})
    return loader.load_snapshot(paths.building.parent / folder)


def _resolve_recipes_path(snapshot) -> Path | None:
    """Resolve the snapshot's dataset key to a local path for hybrid."""
    if snapshot.retrieval_strategy == "dense":
        return None
    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(snapshot.dataset_key)
    return Path(resolved.path)


def _build_pipeline(snapshot, recipes_path: Path | None):
    if snapshot.retrieval_strategy == "dense":
        from src.rag.pipeline.dense_pipeline import DensePipeline
        return DensePipeline(snapshot)
    if snapshot.retrieval_strategy == "hybrid":
        from src.rag.pipeline.hybrid_pipeline import HybridPipeline
        return HybridPipeline(snapshot, recipes_path=str(recipes_path))
    raise ValueError(
        f"Unknown retrieval strategy: {snapshot.retrieval_strategy!r}"
    )


def _retrieve_for(pipeline, strategy):
    """Adapt a pipeline's debug_query to the evaluator's retrieve callable."""
    if strategy == "dense":
        return lambda query, k: [
            r.chunk.recipe_id for r in pipeline.debug_query(query, k)[0]
        ]
    return lambda query, k: [
        r["recipe_id"] for r in pipeline.debug_query(query, k)[0]
    ]


def _load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def _retrieval_params(snapshot) -> dict:
    """Flatten the retrieval-relevant config for MLflow params."""
    cfg = snapshot.base_config
    params = {}
    for key, value in cfg.get("retriever", {}).items():
        params[f"retriever.{key}"] = value
    for key, value in cfg.get("hybrid", {}).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                params[f"hybrid.{key}.{sub_key}"] = sub_value
        else:
            params[f"hybrid.{key}"] = value
    return params


def _print_results(results: dict) -> None:
    for slice_name, res in results.items():
        print(f"\n[{slice_name}] slice mean")
        for metric, value in res["slice"].items():
            print(f"  {metric}: {value:.4f}")
        for query_type, scores in res["query_type"].items():
            print(f"  [{query_type}]")
            for metric, value in scores.items():
                print(f"    {metric}: {value:.4f}")


def main():
    configure_logging()
    args = parse_args()

    folder = _resolve_snapshot_folder(args.snapshot)
    snapshot = _load_snapshot(folder)
    strategy = snapshot.retrieval_strategy
    logger.info("evaluating strategy=%s snapshot=%s", strategy, folder)

    recipes_path = _resolve_recipes_path(snapshot)
    pipeline = _build_pipeline(snapshot, recipes_path)
    retrieve = _retrieve_for(pipeline, strategy)

    evaluator = RetrievalEvaluator(retrieve)
    results = {}
    for slice_name, path in _EVAL_SLICES.items():
        rows = _load_rows(path, limit=args.limit)
        logger.info("evaluating slice=%s rows=%d", slice_name, len(rows))
        results[slice_name] = evaluator.evaluate(rows, ks=tuple(args.ks))

    if args.no_mlflow:
        _print_results(results)
        return

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    run_name = f"eval-{strategy}-{folder}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("logging run %s", run_name)

    with mlflow.start_run(run_name=run_name):
        params = {
            "strategy": strategy,
            "snapshot_name": folder,
            "dataset_key": snapshot.dataset_key,
            "ks": ",".join(str(k) for k in args.ks),
            "eval_files": ",".join(_EVAL_SLICES.values()),
        }
        if args.limit is not None:
            params["limit"] = args.limit
        mlflow.log_params(params)
        mlflow.log_params(_retrieval_params(snapshot))

        for slice_name, res in results.items():
            for metric, value in res["slice"].items():
                mlflow.log_metric(f"{slice_name}.{metric}", value)
            for query_type, scores in res["query_type"].items():
                for metric, value in scores.items():
                    mlflow.log_metric(f"{slice_name}.{query_type}.{metric}", value)

        for path in _EVAL_SLICES.values():
            mlflow.log_artifact(path)

        with tempfile.TemporaryDirectory() as tmp:
            for slice_name, res in results.items():
                out = Path(tmp) / f"results_{slice_name}.json"
                out.write_text(json.dumps(res["per_row"], indent=2))
                mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
