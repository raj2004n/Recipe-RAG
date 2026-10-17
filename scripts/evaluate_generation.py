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
    build_generator,
    build_judge,
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
from src.rag.evaluation.generator_evaluator import GeneratorEvaluator
from src.rag.evaluation.judge_prompts import judge_prompt_hash


logger = logging.getLogger(__name__)

_MLFLOW_EXPERIMENT = "rag-generation-eval"


def parse_args():
    parser = argparse.ArgumentParser()
    return add_common_args(parser).parse_args()


def _print_results(results: dict) -> None:
    print("\n[golden] slice")
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
    logger.info("evaluating generation strategy=%s snapshot=%s", strategy, folder)

    recipes_path = resolve_recipes_path(snapshot)
    recipes_by_id = load_recipes_by_id(recipes_path)

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

    evaluator = GeneratorEvaluator(
        prompt_builder=prompt_builder,
        generator=generator,
        judge=build_judge(judge_cfg),
        recipes_by_id=recipes_by_id,
        judge_calls=judge_cfg["judge_calls"],
    )
    results = evaluator.evaluate(rows)

    if args.no_mlflow:
        _print_results(results)
        return

    run_name = f"eval-gen-{strategy}-{folder}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("logging run %s", run_name)

    with begin_mlflow_run(_MLFLOW_EXPERIMENT, run_name):
        gen_cfg = snapshot.base_config["generator"]
        params = {
            "strategy": strategy,
            "snapshot_name": folder,
            "dataset_key": snapshot.dataset_key,
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
            out = Path(tmp) / "results_golden.json"
            out.write_text(json.dumps(results["per_row"], indent=2))
            mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
