import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.data.pairs.answers import (
    MAX_WORKERS,
    MODEL_ID,
    PROMPTS_JSONL,
    answer_prompts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer every pair prompt with one invented recipe, to be the student's training target.")
    parser.add_argument("--prompts", default=str(PROMPTS_JSONL),
                        help="the prompt set whose rows are answered")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--model", default=MODEL_ID, help="the model that writes the recipes")
    parser.add_argument("--limit", type=int, default=None,
                        help="answer only this many prompts and write to the smoke paths")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent calls")
    parser.add_argument("--out", default=None, help="write the answers here instead")
    parser.add_argument("--resume", action="store_true",
                        help="skip the prompts the output already answers, instead of answering them again")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return answer_prompts(
        prompts_path=Path(args.prompts),
        recipes_override=args.recipes,
        model_id=args.model,
        limit=args.limit,
        output_path=Path(args.out) if args.out is not None else None,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
