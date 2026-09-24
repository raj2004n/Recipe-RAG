"""Shared plumbing for the eval-data draft scripts.

Both ``scripts/draft_prompts.py`` (full fresh draft) and
``scripts/backfill_prompts.py`` (targeted append) need the same encoder
constructor, the same drafter-config kwargs, and the same corpus/output
path resolution. This module is the single source of truth for those bits
so the two scripts can stay focused on their orchestration differences.
"""

from pathlib import Path

from ..core.config import BASE_CONFIG, DATA_PATHS, ENCODER_CONFIG
from ..retrieval.dense.encoder import Encoder


_DRAFTER_CONFIG_KEY = "prompts_drafter"


def build_encoder() -> Encoder:
    """Construct the dense encoder used by both drafter scripts."""
    return Encoder(
        model_id=ENCODER_CONFIG["model_id"],
        batch_size=int(ENCODER_CONFIG["batch_size"]),
        normalize_embeddings=bool(ENCODER_CONFIG["normalize_embeddings"]),
    )


def draft_kwargs(*, include_per_type_count: bool = False) -> dict:
    """Pull model/retry/concurrency knobs from the prompts_drafter config block.

    Set ``include_per_type_count=True`` for the full fresh draft script, which
    needs that knob; the backfill script always derives its per-type count from
    the ``--types`` flag instead.
    """
    cfg = BASE_CONFIG.get(_DRAFTER_CONFIG_KEY, {})
    out: dict = {
        "model_id": cfg.get("model_id", "minimax-m3:cloud"),
        "max_retries": int(cfg.get("max_retries", 3)),
        "backoff_base": float(cfg.get("backoff_base", 2)),
        "max_workers": int(cfg.get("max_workers", 2)),
    }
    if include_per_type_count:
        out["per_type_count"] = int(cfg.get("per_type_count", 8))
    return out


def resolve_paths(output: str | None = None) -> tuple[Path, Path]:
    """Resolve the cleaned-recipes corpus and the prompts.jsonl output path.

    ``output`` (if provided) overrides the default location from
    ``DATA_PATHS["eval_prompts"]``. The output's parent directory is created
    on demand so a fresh repo state can run either script without manual setup.
    """
    corpus_path = Path(DATA_PATHS["cleaned_recipes"])
    prompts_path = Path(output or DATA_PATHS["eval_prompts"])
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    return corpus_path, prompts_path
