"""The argparse surface for the chooser run.

The call loop and the pair set it writes live in src/rag/data/pairs/chooser.py,
which is also where the parser takes its description, so --help prints the
pass's own account of a run rather than a second telling of it. The knobs are the
snapshot the ids resolve in, the smoke and probe switches, where the pairs land,
the worker count and the resume flag.
"""

import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.data.pairs import chooser


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=chooser.__doc__)
    parser.add_argument("--snapshot", default="latest",
                        help="Snapshot target ('latest' or 'current') or folder name.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of primaries; a limited run writes the smoke pair.")
    parser.add_argument("--primaries-file", type=Path, default=None,
                        help="JSON list of primary ids (or an object with 'primary_ids') to run "
                             "instead of the whole pool; writes the probe pair.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where the pairs land, with the manifest beside it as "
                             "<stem>_manifest.json. Use a dated name to keep an earlier set.")
    parser.add_argument("--workers", type=int, default=chooser.MAX_WORKERS,
                        help="Concurrent chooser calls (the provider caps the rate).")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the rows already written and choose only for the missing primaries.")
    parser.add_argument("--probe", action="store_true",
                        help="Make one call, print it whole, and write nothing.")
    args = parser.parse_args()
    if args.probe:
        raise SystemExit(chooser.probe_chooser(snapshot_target=args.snapshot, limit=args.limit,
                                               primaries_file=args.primaries_file))
    raise SystemExit(chooser.choose_pairs(snapshot_target=args.snapshot, limit=args.limit,
                                          primaries_file=args.primaries_file,
                                          output_path=args.out, workers=args.workers,
                                          resume=args.resume))


if __name__ == "__main__":
    main()
