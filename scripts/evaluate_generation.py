"""Evaluate the generation stage against the golden dataset, tracked in MLflow.

Loads a sealed snapshot, builds the serving prompt path and the greedy
generator from it, and answers every golden row through the golden context:
the row's ground-truth recipe_ids resolved against the registered corpus, not
retrieval output. Each generated answer is scored with deterministic
structural checks and a double-called LLM judge. Retrieval is not exercised;
end-to-end evaluation is a separate stage.

Every run logs the snapshot and generator identity, the judge model and
sampling options plus the judge prompt hash, the per-dimension and
per-query-type means, the disagreement rates, and the per-row results as
artifacts.
"""

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import mlflow
from transformers import AutoTokenizer

from src.rag.core.config import BASE_CONFIG, DATA_PATHS
from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import loader, registry
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.core.datasets.resolver import LocalDatasetResolver
from src.rag.data.recipe_format import recipe_to_string
from src.rag.evaluation.generator_evaluator import GeneratorEvaluator, JudgeClient
from src.rag.evaluation.judge_prompts import judge_prompt_hash
from src.rag.generation.generator import Generator
from src.rag.generation.prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)

_MLFLOW_TRACKING_URI = "http://localhost:5000"
_MLFLOW_EXPERIMENT = "rag-generation-eval"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot", type=str, default="latest",
        help="Snapshot target ('latest' or 'current') or folder name to evaluate.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of golden rows evaluated (for smoke runs).",
    )
    parser.add_argument(
        "--probe-judge", action="store_true",
        help="Skip the eval and smoke-test whether the judge provider honors "
             "temperature: five core-judge calls on the first row at temp 0 "
             "and at temp 0.7, reporting how many distinct verdicts come back.",
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


def _resolve_recipes_path(snapshot) -> Path:
    """Resolve the snapshot's dataset key to the registered corpus path."""
    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(snapshot.dataset_key)
    return Path(resolved.path)


def _load_rows(path: str, limit: int | None) -> list[dict]:
    """The verified golden rows: query, recipe_ids, and reference answer."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("answer_verified"):
                    rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _build_prompt_builder(snapshot, tokenizer) -> PromptBuilder:
    return PromptBuilder(
        tokenizer=tokenizer,
        max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
        prompt_templates=snapshot.prompts_config["template"],
    )


def _build_generator(snapshot, tokenizer) -> Generator:
    gen_cfg = snapshot.base_config["generator"]
    return Generator(
        model_id=gen_cfg["model_id"],
        max_input_tokens=gen_cfg["max_input_tokens"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        truncation=gen_cfg["truncation"],
        return_tensors=gen_cfg["return_tensors"],
        adapter_path=gen_cfg["adapter_path"],
        tokenizer=tokenizer,
    )


def _build_judge(judge_cfg: dict) -> JudgeClient:
    return JudgeClient(
        model_id=judge_cfg["model_id"],
        temperature=judge_cfg["temperature"],
        num_ctx=judge_cfg["num_ctx"],
        num_predict=judge_cfg["num_predict"],
        seed=judge_cfg.get("seed"),
        max_retries=judge_cfg["max_retries"],
        backoff_base=judge_cfg["backoff_base"],
    )


def _probe_judge(judge_cfg: dict, row: dict, recipes_by_id: dict) -> None:
    """Smoke-test whether the judge provider honors temperature at all.

    Scores the first golden row's reference answer (a stand-in response, so
    the generator is not needed) five times at temperature 0 and five times
    at a high temperature. A provider that honors the option returns one
    verdict at temp 0; anything else means temperature is being ignored and
    the double-judging disagreement rate would measure provider routing
    nondeterminism instead of judge noise. The seed is varied per call on
    purpose, since a fixed seed would mask exactly the instability this
    probe looks for.
    """
    texts = [
        recipe_to_string(recipes_by_id[rid])
        for rid in row["recipe_ids"]
        if rid in recipes_by_id
    ]
    context = "\n\n".join(texts)
    for temperature in (0.0, 0.7):
        client = JudgeClient(
            model_id=judge_cfg["model_id"],
            temperature=temperature,
            num_ctx=judge_cfg["num_ctx"],
            num_predict=judge_cfg["num_predict"],
            seed=None,
            max_retries=judge_cfg["max_retries"],
            backoff_base=judge_cfg["backoff_base"],
        )
        verdicts = [
            client.score_core(row["query"], context, row["answer"])
            for _ in range(5)
        ]
        distinct = {
            json.dumps(v.model_dump(), sort_keys=True)
            for v in verdicts
            if v is not None
        }
        print(f"probe: temperature={temperature} "
              f"distinct verdicts={len(distinct)}/{len(verdicts)}")
        for blob in sorted(distinct):
            print(f"  {blob}")


def _print_results(results: dict) -> None:
    print("\n[golden] slice")
    for metric, value in results["slice"].items():
        print(f"  {metric}: {value:.4f}")
    for query_type, scores in results["query_type"].items():
        print(f"  [{query_type}]")
        for metric, value in scores.items():
            print(f"    {metric}: {value:.4f}")


def main():
    configure_logging()
    args = parse_args()

    folder = _resolve_snapshot_folder(args.snapshot)
    snapshot = loader.load_snapshot(
        SnapshotPaths.from_paths_config({}).building.parent / folder
    )
    strategy = snapshot.retrieval_strategy
    logger.info("evaluating generation strategy=%s snapshot=%s", strategy, folder)

    recipes_path = _resolve_recipes_path(snapshot)
    with open(recipes_path) as f:
        recipes_by_id = {r["recipe_id"]: r for r in json.load(f)}

    judge_cfg = BASE_CONFIG["generation_judge"]
    rows = _load_rows(DATA_PATHS["eval_answers"], limit=args.limit)
    logger.info("evaluating golden rows=%d", len(rows))

    if args.probe_judge:
        # The probe scores the reference answer as a stand-in response, so it
        # needs neither the tokenizer nor the generator.
        _probe_judge(judge_cfg, rows[0], recipes_by_id)
        return

    gen_cfg = snapshot.base_config["generator"]
    tokenizer = AutoTokenizer.from_pretrained(gen_cfg["model_id"])
    prompt_builder = _build_prompt_builder(snapshot, tokenizer)
    generator = _build_generator(snapshot, tokenizer)

    evaluator = GeneratorEvaluator(
        prompt_builder=prompt_builder,
        generator=generator,
        judge=_build_judge(judge_cfg),
        recipes_by_id=recipes_by_id,
        judge_calls=judge_cfg["judge_calls"],
    )
    results = evaluator.evaluate(rows)

    if args.no_mlflow:
        _print_results(results)
        return

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    run_name = f"eval-gen-{strategy}-{folder}-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info("logging run %s", run_name)

    with mlflow.start_run(run_name=run_name):
        params = {
            "strategy": strategy,
            "snapshot_name": folder,
            "dataset_key": snapshot.dataset_key,
            "generator_model_id": gen_cfg["model_id"],
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