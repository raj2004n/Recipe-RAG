"""CLI wrapper around golden_drafter.draft_candidates.

Reads the model name and concurrency knobs from configs/base_config.yaml
(generation section) and the corpus / output paths from configs/paths_config.yaml.
Supports --limit and --offset so a long draft can be chunked across runs without
re-doing work.
"""

import argparse
from pathlib import Path

from src.rag.core.config import BASE_CONFIG, DATA_PATHS
from src.rag.data.golden_drafter import draft_candidates


def _draft_kwargs() -> dict:
    """Pull model/worker/retry knobs from the golden_drafter section of base_config."""
    gen = BASE_CONFIG.get("golden_drafter", {})
    return {
        "model_name": gen.get("model_id", "gemma3:12b"),
        "max_workers": int(gen.get("max_workers", 4)),
        "max_retries": int(gen.get("max_retries", 3)),
        "backoff_base": int(gen.get("backoff_base", 2)),
    }


def run(args: argparse.Namespace) -> int:
    """Run the drafter with the given args; returns 0 on success."""
    corpus_path = Path(DATA_PATHS["cleaned_recipes"])
    output_path = Path(args.output or DATA_PATHS["eval_golden_candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draft_candidates(
        recipes_path=corpus_path,
        output_path=output_path,
        limit=args.limit,
        offset=args.offset,
        **_draft_kwargs(),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max recipes to draft for in this run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N recipes in the corpus.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output jsonl path. Defaults to data/eval/golden_candidates.jsonl.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
