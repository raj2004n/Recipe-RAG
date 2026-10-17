import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.data.distillation.split import draw_split
from src.rag.data.distillation.tokenise import tokenise_rows
from src.rag.data.distillation.types import (
    ANSWERS_JSONL,
    CATEGORIES_JSONL,
    PAIRS_JSONL,
    SEED,
)
from src.rag.training.student import STUDENT_MODEL_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the distillation split, then write the tokenised rows a trainer reads.")
    parser.add_argument("--run", choices=("split", "tokenise"), default="split",
                        help="draw the split, or tokenise the rows a split names")
    parser.add_argument("--answers", default=str(ANSWERS_JSONL),
                        help="the teacher's answers, whose line order is the row address")
    parser.add_argument("--categories", default=str(CATEGORIES_JSONL),
                        help="the category sidecar the pools are drawn from")
    parser.add_argument("--pairs", default=str(PAIRS_JSONL),
                        help="the pair set the dataset key is read from")
    parser.add_argument("--split-file", default=None,
                        help="tokenise this split; defaults to the newest one written")
    parser.add_argument("--seed", type=int, default=SEED, help="the draw's seed")
    parser.add_argument("--model", default=STUDENT_MODEL_ID, help="the tokenizer's checkpoint")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--out", default=None, help="write the output here instead")
    parser.add_argument("--stamp", default=None, help="date the output with this stamp instead of today")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    if args.run == "split":
        return draw_split(
            answers_path=Path(args.answers),
            categories_path=Path(args.categories),
            seed=args.seed,
            output_path=Path(args.out) if args.out is not None else None,
            stamp=args.stamp,
        )
    return tokenise_rows(
        split_file=args.split_file,
        answers_path=Path(args.answers),
        pairs_path=Path(args.pairs),
        recipes=args.recipes,
        model_id=args.model,
        output_path=Path(args.out) if args.out is not None else None,
        stamp=args.stamp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
