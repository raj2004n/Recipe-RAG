"""The argparse surface for the two pool runs.

The pool pass itself lives in src/rag/data/pairs/pool.py, which is also where the
parser takes its description, so --help prints the pass's own account of a run.
The knobs are the run to make, the snapshot the ids resolve in, the repeat
guard's cut, and a cap on the primaries.
"""

import argparse

from src.rag.core.logging import configure_logging
from src.rag.data.pairs import pool


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=pool.__doc__)
    parser.add_argument("--run", choices=("calibrate", "build"), default="calibrate",
                        help="Calibrate the repeat guard, or write the pool at a given cut.")
    parser.add_argument("--snapshot", default="latest_build",
                        help="Snapshot target ('latest_build') or folder name.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="The repeat guard's cosine cut. Required by --run build.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of primaries (for smoke runs).")
    args = parser.parse_args()
    if args.run == "build" and args.threshold is None:
        parser.error("--run build needs --threshold")
    if args.run == "calibrate":
        raise SystemExit(pool.calibrate_guard(snapshot_target=args.snapshot, limit=args.limit))
    raise SystemExit(pool.build_pool(snapshot_target=args.snapshot, limit=args.limit,
                                     threshold=args.threshold))


if __name__ == "__main__":
    main()
