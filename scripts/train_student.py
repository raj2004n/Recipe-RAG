"""Distil the teacher's recipe answers into the student generator.

The argparse surface, the default paths and the exit code live here; the
training loop itself lives in src/rag/training/student.py. The knobs are the
tokenised rows to read, the directory a run writes its adapters into, the
epochs and rows it covers, the seed, and the MLflow switch.
"""

import argparse
from pathlib import Path

from src.rag.core.logging import configure_logging
from src.rag.training import train_student
from src.rag.training.student import ARTIFACTS_DIR, EPOCHS, SEED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default=None,
                        help="train on these tokenised rows; defaults to the newest file written")
    parser.add_argument("--out-dir", default=str(ARTIFACTS_DIR),
                        help="write the run's adapters under this directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="passes over the training rows")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the training rows an epoch draws, for smoke runs")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="the seed the row order and the dropout share")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip MLflow logging; the adapters and their manifests are still written")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return train_student(
        rows_path=Path(args.rows) if args.rows is not None else None,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        limit=args.limit,
        seed=args.seed,
        no_mlflow=args.no_mlflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
