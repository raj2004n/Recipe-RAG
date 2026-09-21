"""Rank student adapters by having the judge read what they write.

The argparse surface, the default paths and the exit code live here; the sweep
itself lives in src/rag/evaluation/adapter_judge.py. The knobs are the split to
read, the adapters to compare, the directory a run lands in, the probe switch
and the threads the judge calls run on.
"""

import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.evaluation.adapter_judge import (
    ARTIFACTS_DIR,
    DEFAULT_SYSTEMS,
    PROBE_ROWS,
    judge_adapters,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev",
                        help="which tokenised rows to read (default: dev)")
    parser.add_argument("--systems", default=DEFAULT_SYSTEMS,
                        help="comma-separated adapter folder names, or 'base' for none")
    parser.add_argument("--out-dir", default=str(ARTIFACTS_DIR),
                        help="where the run folder is written")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the rows read, for a smoke run")
    parser.add_argument("--probe", action="store_true",
                        help="time generation on a few rows and write nothing")
    parser.add_argument("--probe-rows", type=int, default=PROBE_ROWS,
                        help="how many rows the probe measures")
    parser.add_argument("--judge-workers", type=int, default=4,
                        help="threads used for the judge calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip the MLflow run and keep the printed table")
    args = parser.parse_args(argv)
    args.systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not args.systems:
        raise SystemExit("at least one system is needed")
    return args


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return judge_adapters(
        split=args.split,
        systems=args.systems,
        out_dir=Path(args.out_dir),
        limit=args.limit,
        probe=args.probe,
        probe_rows=args.probe_rows,
        judge_workers=args.judge_workers,
        seed=args.seed,
        no_mlflow=args.no_mlflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
