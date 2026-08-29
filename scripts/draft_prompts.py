"""CLI wrapper around prompts_drafter.draft_prompts.

Reads the model and concurrency knobs from configs/base_config.yaml
(prompts_drafter section) and the corpus / output paths from
configs/paths_config.yaml. Supports --per-type-count so a long draft can be
run as a small smoke test before committing to the full 88 rows.
"""

import argparse

from _draft_common import build_encoder, draft_kwargs, resolve_paths
from src.rag.data.prompts_drafter import draft_prompts


def run(args: argparse.Namespace) -> int:
    """Run the drafter with the given args; returns 0 on success."""
    corpus_path, output_path = resolve_paths(args)
    kwargs = draft_kwargs(include_per_type_count=True)
    if args.per_type_count is not None:
        kwargs["per_type_count"] = args.per_type_count
    if args.seed is not None:
        kwargs["seed"] = args.seed
    draft_prompts(
        recipes_path=corpus_path,
        output_path=output_path,
        encoder=build_encoder(),
        **kwargs,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-type-count",
        type=int,
        default=None,
        help="Rows per query type. Defaults to the prompts_drafter.per_type_count config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the selection RNG. Same seed + same corpus => same selections.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output jsonl path. Defaults to data/eval/prompts.jsonl.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
