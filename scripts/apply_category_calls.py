"""Apply the reviewed category calls to the recipe label sidecar.

The write itself lives in src/rag/data/pairs/categories.py, beside the labelling
pass that produced the sidecar it edits. The knobs are the sidecar to write into,
the manifest to record the calls in, and the dry run that prints the calls and
the resulting counts without touching either file.
"""

import argparse
from pathlib import Path

from src.rag.data.pairs.categories import (
    SIDECAR_JSONL,
    SIDECAR_MANIFEST,
    apply_category_calls,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", default=str(SIDECAR_JSONL),
                        help="the label sidecar to write the calls into")
    parser.add_argument("--manifest", default=str(SIDECAR_MANIFEST),
                        help="the manifest to record the calls in")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the calls and the resulting counts, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return apply_category_calls(
        sidecar_path=Path(args.sidecar),
        manifest_path=Path(args.manifest),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
