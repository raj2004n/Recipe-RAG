import argparse

from src.core.logging import configure_logging
from src.pairs import pool


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="""Build the candidate pool the chooser reads, and calibrate its repeat guard.

The pool is the build-time half of pair selection. For every recipe in the
registered corpus it stores the top-ranked neighbours in the whole-recipe
embedding space, so the chooser can be handed a primary's candidates without
anything being re-encoded. Ranks 10 to 30 are the band a partner is drawn from,
ranks 1 to 3 are the plausibility anchor the chooser sees for a primary whose
band holds nothing workable, and ranks 4 to 9 are kept only so the band's edge
can be re-derived later.

The repeat guard is a cosine cut against the primary's own vector, and a flagged
candidate is never offered to the chooser: it is left out of the band, and the
fallback anchor is the closest candidate that is not flagged. The rows stay in
the pool so the flagging can be audited after the fact.

Two runs. `calibrate` measures what a cut flags across the pool and prints its
reach, writing nothing. `build` writes the pool and its manifest at a cut passed
on the command line, so the guard's value is chosen with its consequences in
front of it rather than buried in the code.

The corpus is read through the snapshot's registered dataset key and is never
written to. It is the only input, and the pair set this pool feeds stays
independent of every other dataset.
""")
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
