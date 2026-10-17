import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import mlflow

from src.rag.core.config import DATA_PATHS
from src.rag.core.logging import configure_logging
from src.rag.evaluation.end_to_end_evaluator import EndToEndEvaluator
from src.rag.evaluation.harness import (
    add_common_args,
    begin_mlflow_run,
    build_generator,
    build_judge,
    build_pipeline,
    build_prompt_builder,
    judge_config,
    load_recipes_by_id,
    load_rows,
    load_snapshot,
    print_scores,
    probe_judge,
    resolve_recipes_path,
    resolve_snapshot_folder,
    tokenizer_for,
)
from src.rag.evaluation.judge_prompts import judge_prompt_hash


logger = logging.getLogger(__name__)

_MLFLOW_EXPERIMENT = "rag-end-to-end-eval"

# Dense serving answers the top-k retriever hits with no reranking stage.
_DENSE_SERVING_K = 3


def parse_args():
    parser = argparse.ArgumentParser()
    return add_common_args(parser).parse_args()


def _serving_k(snapshot) -> int:
    """The context window size the pipeline serves: k for dense, final_top_k
    for hybrid, which is already deduped to whole recipes."""
    if snapshot.retrieval_strategy == "hybrid":
        return snapshot.base_config["hybrid"]["prompt"]["final_top_k"]
    return _DENSE_SERVING_K


def _retrieve_recipes(pipeline, snapshot, recipes_by_id):
    """A retrieve callable serving whole recipes in rank order for the strategy.

    Hybrid's debug_query already resolves chunks to whole recipes; dense
    returns RetrievalResult objects, which are resolved back to recipes
    through the registered corpus so the two strategies feed the evaluator,
    and therefore the judge context and the prompt, identical shapes.
    """
    if snapshot.retrieval_strategy == "hybrid":
        return lambda query, k: pipeline.debug_query(query, k)[0]

    def retrieve(query, k):
        return [
            recipes_by_id[r.chunk.recipe_id]
            for r in pipeline.debug_query(query, k)[0]
            if r.chunk.recipe_id in recipes_by_id
        ]

    return retrieve


def _print_results(results: dict) -> None:
    print("\n[e2e] slice")
    print_scores(results["slice"])
    for query_type, scores in results["query_type"].items():
        print(f"  [{query_type}]")
        print_scores(scores, indent="    ")


def main():
    configure_logging()
    args = parse_args()

    folder = resolve_snapshot_folder(args.snapshot)
    snapshot = load_snapshot(folder)
    strategy = snapshot.retrieval_strategy
    k = _serving_k(snapshot)
    logger.info("evaluating end-to-end strategy=%s snapshot=%s k=%d", strategy, folder, k)

    recipes_path = resolve_recipes_path(snapshot)
    recipes_by_id = load_recipes_by_id(recipes_path)
    pipeline = build_pipeline(snapshot, recipes_path)
    retrieve = _retrieve_recipes(pipeline, snapshot, recipes_by_id)

    judge_cfg = judge_config()
    rows = load_rows(DATA_PATHS["eval_answers"], limit=args.limit, verified=True)
    logger.info("evaluating golden rows=%d", len(rows))

    if args.probe_judge:
        # The probe scores the reference answer as a stand-in response, so it
        # needs neither the tokenizer nor the generator.
        probe_judge(judge_cfg, rows[0], recipes_by_id)
        return

    tokenizer = tokenizer_for(snapshot)
    prompt_builder = build_prompt_builder(snapshot, tokenizer)
    generator = build_generator(snapshot, tokenizer)

    evaluator = EndToEndEvaluator(
        prompt_builder=prompt_builder,
        generator=generator,
        judge=build_judge(judge_cfg),
        retrieve=retrieve,
        k=k,
        judge_calls=judge_cfg["judge_calls"],
    )
    results = evaluator.evaluate(rows)

    if args.no_mlflow:
        _print_results(results)
        return

    run_name = f"eval-e2e-{strategy}-{folder}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("logging run %s", run_name)

    with begin_mlflow_run(_MLFLOW_EXPERIMENT, run_name):
        gen_cfg = snapshot.base_config["generator"]
        params = {
            "strategy": strategy,
            "snapshot_name": folder,
            "dataset_key": snapshot.dataset_key,
            "serving_k": k,
            "generator_model_id": gen_cfg["model_id"],
            "generator_adapter_path": gen_cfg.get("adapter_path") or "",
            "judge_model_id": judge_cfg["model_id"],
            "judge_temperature": judge_cfg["temperature"],
            "judge_calls": judge_cfg["judge_calls"],
            "judge_prompt_hash": judge_prompt_hash(),
            "rows": len(rows),
            "eval_file": DATA_PATHS["eval_answers"],
        }
        if judge_cfg.get("seed") is not None:
            params["judge_seed"] = judge_cfg["seed"]
        if args.limit is not None:
            params["limit"] = args.limit
        mlflow.log_params(params)

        for metric, value in results["slice"].items():
            mlflow.log_metric(metric, value)
        for query_type, scores in results["query_type"].items():
            for metric, value in scores.items():
                mlflow.log_metric(f"{query_type}.{metric}", value)

        mlflow.log_artifact(DATA_PATHS["eval_answers"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results_e2e.json"
            out.write_text(json.dumps(results["per_row"], indent=2))
            mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
