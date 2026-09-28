"""Shared snapshot, pipeline, and judge plumbing for the evaluation scripts.

Every evaluation run follows the same shape: resolve a sealed snapshot, build
the components for its strategy, load the rows, run an evaluator, then log to
MLflow. The helpers here are the single definition of that resolution and
construction, so the scripts stay thin and never drift on how a snapshot
target, a corpus path, a pipeline, or the judge client is built. The pair
builders and the student training run draw on the same plumbing, which is what
keeps them reading a snapshot exactly the way an evaluation does.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
from transformers import AutoTokenizer

from ..core.config import BASE_CONFIG
from ..core.datasets.resolver import LocalDatasetResolver
from ..core.snapshot import loader, registry
from ..core.snapshot.paths import SnapshotPaths
from ..data.recipe_format import recipe_to_string
from ..generation.generator import Generator
from ..generation.prompt_builder import PromptBuilder
from .generator_evaluator import JudgeClient


logger = logging.getLogger(__name__)

# The scripts write straight to the local SQLite store by default, so no MLflow
# server needs to be running to log or read runs. Point RECIPE_RAG_MLFLOW_TRACKING_URI
# at an HTTP URI only when you want a run to go through a server (for example to
# watch it live in the UI).
MLFLOW_TRACKING_URI = os.environ.get("RECIPE_RAG_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
_DEFAULT_SNAPSHOT = "latest_build"


def add_common_args(parser: argparse.ArgumentParser, *, with_probe: bool = True):
    """The snapshot, limit, and MLflow args every eval script shares."""
    parser.add_argument(
        "--snapshot", type=str, default=_DEFAULT_SNAPSHOT,
        help="Snapshot target ('latest_build' or 'build_in_progress') or "
             "folder name to evaluate.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of rows evaluated (for smoke runs).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Skip MLflow logging; print the results instead.",
    )
    if with_probe:
        parser.add_argument(
            "--probe-judge", action="store_true",
            help="Skip the eval and smoke-test whether the judge provider honors "
                 "temperature: five core-judge calls on the first row at temp 0 "
                 "and at temp 0.7, reporting how many distinct verdicts come back.",
        )
    return parser


def resolve_snapshot_folder(target: str) -> str:
    """Resolve a target name or folder name to a snapshot folder name."""
    if target in ("latest_build", "build_in_progress"):
        paths = SnapshotPaths.from_paths_config({})
        folder = registry.get_folder_for_target(paths.registry_file, target)
        if not folder:
            raise FileNotFoundError(
                f"No snapshot folder mapped to target {target!r} in registry"
            )
        return folder
    return target


def load_snapshot(folder: str):
    """Load the sealed snapshot in the given folder, validating its status."""
    paths = SnapshotPaths.from_paths_config({})
    return loader.load_snapshot(paths.building.parent / folder)


def resolve_recipes_path(snapshot) -> Path:
    """Resolve the snapshot's dataset key to the registered corpus path."""
    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(snapshot.dataset_key)
    return Path(resolved.path)


def load_rows(path: str, limit: int | None, *, verified: bool = False) -> list[dict]:
    """The jsonl rows for a slice, optionally only the verified ones.

    The generation and end-to-end evals score verified golden rows, which carry
    a reference answer; the retrieval eval scores every row in a slice, verified
    or archived, none of which carry the verified flag. verified=True therefore
    keeps only rows whose answer_verified field is set.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if verified and not row.get("answer_verified"):
                continue
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_recipes_by_id(path: Path) -> dict[int, dict]:
    """The corpus recipes keyed by recipe_id, the snapshot's retrieval ground truth."""
    with open(path) as f:
        return {r["recipe_id"]: r for r in json.load(f)}


def build_pipeline(snapshot, recipes_path: Path | None = None):
    """A pipeline for the snapshot's strategy, with the recipes source threaded.

    Dense pipelines need only the snapshot; hybrid pipelines need the registered
    corpus path to resolve reranked chunks back to whole recipes.
    """
    from ..pipeline.dense_pipeline import DensePipeline
    from ..pipeline.hybrid_pipeline import HybridPipeline

    strategy = snapshot.retrieval_strategy
    if strategy == "dense":
        return DensePipeline(snapshot)
    if strategy == "hybrid":
        if recipes_path is None:
            raise ValueError("hybrid evaluation needs the recipes path")
        return HybridPipeline(snapshot, recipes_path=str(recipes_path))
    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")


def build_prompt_builder(snapshot, tokenizer) -> PromptBuilder:
    return PromptBuilder(
        tokenizer=tokenizer,
        max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
        prompt_templates=snapshot.prompts_config["template"],
    )


def build_generator(snapshot, tokenizer) -> Generator:
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


def build_judge(judge_cfg: dict) -> JudgeClient:
    return JudgeClient(
        model_id=judge_cfg["model_id"],
        temperature=judge_cfg["temperature"],
        num_ctx=judge_cfg["num_ctx"],
        num_predict=judge_cfg["num_predict"],
        seed=judge_cfg.get("seed"),
        max_retries=judge_cfg["max_retries"],
        backoff_base=judge_cfg["backoff_base"],
    )


def judge_config() -> dict:
    return BASE_CONFIG["generation_judge"]


def probe_judge(judge_cfg: dict, row: dict, recipes_by_id: dict) -> None:
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


def print_scores(scores: dict, indent: str = "  ") -> None:
    """Pretty-print a flat {metric: value} score dict."""
    for metric, value in scores.items():
        print(f"{indent}{metric}: {value:.4f}")


def begin_mlflow_run(experiment: str, run_name: str):
    """Open an MLflow run in the shared tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def tokenizer_for(snapshot):
    """The AutoTokenizer for the snapshot's generator model."""
    return AutoTokenizer.from_pretrained(snapshot.base_config["generator"]["model_id"])
