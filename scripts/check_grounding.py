import argparse
from pathlib import Path

from src.core.logging import configure_logging
from src.pairs.types import PAIRS_JSONL
from src.evaluation.grounding import DEFAULT_PROMPTS, check_grounding


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag prompts that ask for a way of cooking the pair gives the teacher no basis for.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS),
                        help="the designed prompts to measure")
    parser.add_argument("--pairs", default=str(PAIRS_JSONL),
                        help="the pair set the prompts were designed from")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--out", default=None,
                        help="write the annotated rows here, default beside the prompts file")
    parser.add_argument("--no-write", action="store_true", help="report without writing a file")
    parser.add_argument("--show", type=int, default=15,
                        help="how many techniques and rows to print in full")
    parser.add_argument("--limit", type=int, default=None, help="measure only this many rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return check_grounding(
        prompts_path=Path(args.prompts),
        pairs_path=Path(args.pairs),
        recipes_override=args.recipes,
        out_path=Path(args.out) if args.out is not None else None,
        no_write=args.no_write,
        show=args.show,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
