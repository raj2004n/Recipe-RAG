"""CLI wrapper around prompts_drafter.draft_prompts.

Reads the model and concurrency knobs from configs/base_config.yaml
(prompts_drafter section) and the corpus / output paths from
configs/paths_config.yaml. Supports --per-type-count so a long draft can be
run as a small smoke test before committing to the full 72 rows.
"""

import argparse
from pathlib import Path

from src.rag.core.config import BASE_CONFIG, DATA_PATHS, ENCODER_CONFIG
from src.rag.data.prompts_drafter import draft_prompts
from src.rag.retrieval.dense.encoder import Encoder


def _draft_kwargs() -> dict:
    """Pull model/worker/retry knobs from the prompts_drafter section of base_config."""
    cfg = BASE_CONFIG.get("prompts_drafter", {})
    return {
        "model_id": cfg.get("model_id", "minimax-m3:cloud"),
        "max_retries": int(cfg.get("max_retries", 3)),
        "backoff_base": float(cfg.get("backoff_base", 2)),
        "max_workers": int(cfg.get("max_workers", 2)),
        "per_type_count": int(cfg.get("per_type_count", 8)),
    }


def _encoder() -> Encoder:
    return Encoder(
        model_id=ENCODER_CONFIG["model_id"],
        batch_size=int(ENCODER_CONFIG["batch_size"]),
        normalize_embeddings=bool(ENCODER_CONFIG["normalize_embeddings"]),
    )


def run(args: argparse.Namespace) -> int:
    """Run the drafter with the given args; returns 0 on success."""
    corpus_path = Path(DATA_PATHS["cleaned_recipes"])
    output_path = Path(args.output or DATA_PATHS["eval_prompts"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = _draft_kwargs()
    if args.per_type_count is not None:
        kwargs["per_type_count"] = args.per_type_count
    if args.seed is not None:
        kwargs["seed"] = args.seed
    draft_prompts(
        recipes_path=corpus_path,
        output_path=output_path,
        encoder=_encoder(),
        **kwargs,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-type-count",
        type=int,
        default=None,
        help="Rows per query type. Defaults to the prompts_drafter.per_type_count config value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the selection RNG. Same seed + same corpus => same selections.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output jsonl path. Defaults to data/eval/prompts.jsonl.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
