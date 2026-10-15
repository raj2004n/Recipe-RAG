"""Evaluate retrieval for a snapshot against the eval datasets, tracked in MLflow.

Loads a sealed snapshot, builds the pipeline for its strategy, and scores
retrieval over the slices `_EVAL_SLICES` names. Each run logs the snapshot
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
from src.rag.evaluation.harness import (
    add_common_args,
    begin_mlflow_run,
    build_pipeline,
    load_rows,
    load_snapshot,
    print_scores,
    resolve_recipes_path,
    resolve_snapshot_folder,
)
from src.rag.evaluation.retriever_evaluator import RetrievalEvaluator


logger = logging.getLogger(__name__)

_MLFLOW_EXPERIMENT = "rag-retrieval-eval"

# Slice name -> eval file. A slice is scored and reported on its own, never
# blended into another.
_EVAL_SLICES = {
    "golden": DATA_PATHS["eval_golden_prompts"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser, with_probe=False)
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10],
        help="k values to score. Defaults to 1 2 3 5 10.",
    )
    return parser.parse_args()



def _retrieve_for(pipeline, strategy):
    """Adapt a pipeline's debug_query to the evaluator's retrieve callable."""
    if strategy == "dense":
        return lambda query, k: [
            r.chunk.recipe_id for r in pipeline.debug_query(query, k)[0]
        ]
    return lambda query, k: [
        r["recipe_id"] for r in pipeline.debug_query(query, k)[0]
    ]


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
        print_scores(res["slice"])
        for query_type, scores in res["query_type"].items():
            print(f"  [{query_type}]")
            print_scores(scores, indent="    ")


def main():
    configure_logging()
    args = parse_args()

    folder = resolve_snapshot_folder(args.snapshot)
    snapshot = load_snapshot(folder)
    strategy = snapshot.retrieval_strategy
    logger.info("evaluating strategy=%s snapshot=%s", strategy, folder)

    recipes_path = (
        resolve_recipes_path(snapshot) if strategy == "hybrid" else None
    )
    pipeline = build_pipeline(snapshot, recipes_path)
    retrieve = _retrieve_for(pipeline, strategy)

    evaluator = RetrievalEvaluator(retrieve)
    results = {}
    for slice_name, path in _EVAL_SLICES.items():
        rows = load_rows(path, limit=args.limit)
        logger.info("evaluating slice=%s rows=%d", slice_name, len(rows))
        results[slice_name] = evaluator.evaluate(rows, ks=tuple(args.ks))

    if args.no_mlflow:
        _print_results(results)
        return

    run_name = f"eval-{strategy}-{folder}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("logging run %s", run_name)

    with begin_mlflow_run(_MLFLOW_EXPERIMENT, run_name):
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
