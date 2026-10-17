import argparse
from pathlib import Path

from src.core.logging import configure_logging
from src.pairs.design import design_prompts
from src.pairs.types import MAX_WORKERS, MODEL_ID, PAIRS_JSONL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design one writing prompt per recipe pair, for a teacher model to answer.")
    parser.add_argument("--pairs", default=str(PAIRS_JSONL),
                        help="the pair set whose rows are designed from")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--model", default=MODEL_ID, help="the model that writes the prompts")
    parser.add_argument("--limit", type=int, default=None,
                        help="design only this many pairs and write to the smoke paths")
    parser.add_argument("--primaries-file", default=None,
                        help="a JSON list of primary ids to design, written to the probe paths")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent calls")
    parser.add_argument("--out", default=None, help="write the prompts here instead")
    parser.add_argument("--resume", action="store_true",
                        help="keep the rows already in the output and design only the rest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return design_prompts(
        Path(args.pairs),
        recipes_override=args.recipes,
        model_id=args.model,
        limit=args.limit,
        primaries_file=args.primaries_file,
        output_path=Path(args.out) if args.out is not None else None,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
