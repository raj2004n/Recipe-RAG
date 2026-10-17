import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.data.pairs import chooser


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="""Run the chooser over the candidate pool and write the pair set.

One call per primary, and each call returns two picks, so every primary appears in
two pairs with a different partner each time. The chooser is shown the primary
recipe in full, the band of candidates its partners must come from, and the anchor
ranks it may fall back to, and it returns two ranks with a one-sentence reason
each. Those reasons are the audit trail for the rows, and the pair set is those
rows: the two recipe ids, the partner's rank and cosine, the anchors it was
measured against, the flag and the reason, keyed to the snapshot and dataset the
ids resolve in. Each row names the other pick from the same call, so a pair can be
read either alone or as one half of a joint decision.

The band and the anchors both arrive with the repeat guard already applied, so a
candidate the guard flagged is never offered and a flagged anchor cannot be
returned. What the model reads is therefore the pool's own decision, not a
second opinion about it.

The calls are the only spend in this task: the pool is local, and nothing here
generates a recipe, writes a prompt for the teacher, or touches any dataset
other than the registered corpus the pool was built from.

A run with --limit writes to the smoke paths and prints every reason, which is
how the prompt gets read before the corpus-wide run pays for it.
""")
    parser.add_argument("--snapshot", default="latest_build",
                        help="Snapshot target ('latest_build') or folder name.")
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
