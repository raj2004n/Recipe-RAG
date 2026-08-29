"""CLI wrapper around answers_drafter.draft_answers.

Reads the model and concurrency knobs from
``configs/base_config.yaml`` (``answers_drafter`` section) and the
prompts / recipes / output paths from ``configs/paths_config.yaml``.
Supports ``--per-type-count`` so a long draft can be run as a small
smoke test before committing to the full 104 rows; ``--per-type-count 1``
produces 13 rows total (one per query type).
"""

import argparse
import random
from pathlib import Path

from src.rag.core.config import BASE_CONFIG, DATA_PATHS
from src.rag.data.answers_drafter import draft_answers
from src.rag.data.answers_drafter.selection import load_prompts
from src.rag.data.answers_drafter.types import QUERY_TYPES
from src.rag.data.prompts_drafter.selection import load_recipes


_DRAFTER_CONFIG_KEY = "answers_drafter"


def _draft_kwargs() -> dict:
    """Pull model/retry/concurrency knobs from the answers_drafter config block."""
    cfg = BASE_CONFIG.get(_DRAFTER_CONFIG_KEY, {})
    return {
        "model_id": cfg.get("model_id", "minimax-m3:cloud"),
        "max_retries": int(cfg.get("max_retries", 3)),
        "backoff_base": float(cfg.get("backoff_base", 2)),
        "max_workers": int(cfg.get("max_workers", 2)),
    }


def _resolve_paths(args) -> tuple[Path, Path, Path]:
    """Resolve prompts, recipes, and output paths from config + CLI overrides."""
    recipes_path = Path(DATA_PATHS["cleaned_recipes"])
    prompts_path = Path(args.prompts or DATA_PATHS["eval_prompts"])
    output_path = Path(args.output or DATA_PATHS["eval_answers"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return recipes_path, prompts_path, output_path


def _filter_prompts_by_per_type_count(
    prompts_path: Path, per_type_count: int | None, seed: int | None,
) -> Path:
    """Write a sliced prompts file with at most per_type_count rows per type.

    Returns the path to the sliced file. The original prompts file is
    untouched. If ``per_type_count`` is None, returns ``prompts_path``
    unchanged. The slice is deterministic given the same seed so a
    smoke-test rerun produces the same 13 rows.
    """
    if per_type_count is None:
        return prompts_path
    rows = load_prompts(prompts_path)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for qt in QUERY_TYPES:
        of_type = [r for r in rows if r["query_type"] == qt]
        rng.shuffle(of_type)
        sampled.extend(of_type[:per_type_count])
    sliced_path = prompts_path.with_name(
        f"{prompts_path.stem}.slice{per_type_count}.jsonl"
    )
    with sliced_path.open("w") as f:
        for row in sampled:
            f.write(__import__("json").dumps(row) + "\n")
    print(f"Sliced {len(sampled)} prompts (per_type_count={per_type_count}) to {sliced_path}")
    return sliced_path


def run(args: argparse.Namespace) -> int:
    """Run the drafter with the given args; returns 0 on success."""
    recipes_path, prompts_path, output_path = _resolve_paths(args)
    sliced_prompts = _filter_prompts_by_per_type_count(
        prompts_path, args.per_type_count, args.seed,
    )
    draft_answers(
        prompts_path=sliced_prompts,
        recipes_path=recipes_path,
        output_path=output_path,
        **_draft_kwargs(),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-type-count",
        type=int,
        default=None,
        help=(
            "Rows per query type. Defaults to no slicing (all rows in the "
            "prompts file are drafted). Use 1 to smoke-test with 13 rows "
            "(one per query type) before committing to the full 104."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for the smoke-test slice RNG. Same seed + same prompts "
            "file => same 13 rows."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Override the input prompts jsonl path. Defaults to data/eval/prompts.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output jsonl path. Defaults to data/eval/golden_answers.jsonl.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
