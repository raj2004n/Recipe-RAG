import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.data.pairs.categories import (
    INPUT_JSONL,
    MAX_WORKERS,
    MODEL_ID,
    label_categories,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label every recipe in the pair set as savoury or sweet, from its title.")
    parser.add_argument("--input", default=str(INPUT_JSONL),
                        help="the pair set whose recipe ids are labelled")
    parser.add_argument("--model", default=MODEL_ID, help="the model that makes the call")
    parser.add_argument("--limit", type=int, default=None,
                        help="label only this many titles and write to the smoke paths")
    parser.add_argument("--ids-file", default=None,
                        help="a JSON list of recipe ids to label, written to the probe paths")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent calls")
    parser.add_argument("--out", default=None, help="write the labels here instead")
    parser.add_argument("--resume", action="store_true",
                        help="keep rows already in the output and label the rest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return label_categories(
        input_path=Path(args.input),
        model_id=args.model,
        limit=args.limit,
        ids_file=args.ids_file,
        output_path=Path(args.out) if args.out is not None else None,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
