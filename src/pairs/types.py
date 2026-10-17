import re
from datetime import datetime, timezone
from pathlib import Path

# The pair set this reads and where the designed prompts land. A pass writes to a dated file, so
# the next pass sits beside it rather than on top of it. A limited run writes the smoke files
# instead, which are a scratch surface and are overwritten on purpose.
PAIRS_JSONL = Path("data/pairs/recipe_pairs_20260918.jsonl")
OUTPUT_DIR = Path("data/pairs")
SMOKE_JSONL = Path("data/pairs/recipe_prompts_smoke.jsonl")
SMOKE_MANIFEST = Path("data/pairs/recipe_prompts_smoke_manifest.json")

# A run over named primaries writes here. It exists so pairs whose prompt needs a second look can
# be redesigned on their own, without disturbing the smoke slice or the full set.
PROBE_JSONL = Path("data/pairs/recipe_prompts_probe.jsonl")
PROBE_MANIFEST = Path("data/pairs/recipe_prompts_probe_manifest.json")

# The design call: greedy, seeded, and thinking off. The call reads two recipes and writes one
# request, which is a composition rather than a derivation, and the measured chooser call spent
# roughly ten times the answer on reasoning this call does not need.
MODEL_ID = "qwen3.5:397b-cloud"
TEMPERATURE = 0.0
SEED = 42
THINK = False
NUM_CTX = 16384
NUM_PREDICT = 1200

# Sampling retries. A call that fails to parse or validate is retried with backoff; a pair whose
# retries all fail is reported in the manifest rather than dropped from the set.
MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_WORKERS = 3

# The version of the instructions these rows were designed under. Changing them means changing
# this, so rows written under the old instructions stay identifiable.
METHOD = "pair-prompt-v1"

# The word band a prompt is expected to land in, and the patterns a read-through would raise a
# flag about. Flags never drop a row: they sort the review list. A time is deliberately not an
# amount, because a constraint like "twenty minutes" is one of the moves the instructions invite.
WORDS_MIN = 15
WORDS_MAX = 90
QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:g|kg|ml|l|cups?|tbsps?|tsps?|oz|lb|grams?|°c|°f)\b",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"(?:^|\s)\d+[.)]\s")
META_PHRASES = (
    "the pair", "the two recipes above", "recipe 1", "recipe 2", "your task",
    "the rubric", "a novel recipe", "the following recipes", "as an ai",
)
# Title words that carry no dish identity, so a shared one is not a sign of copying.
TITLE_STOPWORDS = {
    "and", "with", "the", "a", "an", "of", "for", "to", "in", "on", "or", "my", "your",
    "easy", "best", "quick", "simple", "recipe", "homemade", "style", "way", "how",
}


def dated_output_paths(stamp: str | None = None) -> tuple[Path, Path]:
    """This pass's prompts and manifest, dated so a later pass does not land on them."""
    day = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"recipe_prompts_{day}.jsonl"
    return out_path, out_path.with_name(f"{out_path.stem}_manifest.json")
