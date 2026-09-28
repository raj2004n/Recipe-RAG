"""Metadata for the distillation passes: the paths, the draw, and the caps.

Both passes read the same files, so they share one idea of where the answers, the
category sidecar and the pair set live, and where their own output lands. The draw
sizes and the expected counts are the numbers the split was priced at, and they are
checked rather than trusted, so a split that is not the one settled stops before it
is written. The caps say how long a row may be and how many rows are held in hand at
once.

Nothing here touches a file or a model, so a caller can import it for the constants
alone.
"""

from datetime import datetime, timezone
from pathlib import Path

# The answers the teacher wrote and the category sidecar that sorts their primaries. The answers
# file's own line order is the row address the split and the tokenised rows are keyed by.
ANSWERS_JSONL = Path("data/pairs/recipe_answers_20260918.jsonl")
CATEGORIES_JSONL = Path("data/pairs/recipe_categories.jsonl")
PAIRS_JSONL = Path("data/pairs/recipe_pairs_20260918.jsonl")
OUTPUT_DIR = Path("data/pairs")

# The draw's seed. The split is a function of this number and the corpus, so the manifest is enough
# to rebuild it.
SEED = 42

# How many rows each pool gives to the eval. The three sizes are what makes the eval 297 savoury
# against 297 sweet, and they were settled before anything was written.
EVAL_DRAW = {
    "savoury_primary_only": 297,
    "sweet_primary_only": 74,
    "sweet_with_partners": 223,
}

# The dev slice, drawn from the training rows by primary category so it mirrors them.
DEV_DRAW = {"savoury": 186, "sweet": 14}

# Every count the split has to land on, checked rather than trusted.
EXPECTED_POOLS = {
    "savoury_primary_only": 908,
    "savoury_with_partners": 1586,
    "sweet_primary_only": 148,
    "sweet_with_partners": 324,
}
EXPECTED_SPLIT = {"train": 2172, "dev": 200, "eval": 594}
EXPECTED_CATEGORY = {
    "train": {"savoury": 2011, "sweet": 161},
    "dev": {"savoury": 186, "sweet": 14},
    "eval": {"savoury": 297, "sweet": 297},
}

# The sequence cap. The longest row in the corpus is 3,964 tokens, so nothing is truncated and the
# cap is a guard against a render that has gone wrong rather than a limit on the data.
MAX_SEQ_LEN = 4096

# Rows are tokenised in blocks, so a corpus sized pass never holds every id list in Python objects
# at once.
TOKENISE_BATCH = 256

# How a row's primary is judged for the leak figures. Both readings are reported, because the split
# was priced against the first one and the second is the one that decides how much of the eval is
# really unseen.
LEAK_RULE = (
    "a row's partner is trained when that recipe appears in a training row on either side. A row's "
    "primary is absent from the partner slot when that recipe appears in no training row's partner "
    "position, which is the reading the split was priced against, and it is separately counted as "
    "present in training whenever that recipe appears in any training row at all, which is what the "
    "model has actually read about the dish"
)


def dated_paths(stem: str, suffix: str, stamp: str | None = None) -> tuple[Path, Path]:
    """This pass's output and its manifest, dated so a later pass lands beside it."""
    day = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"{stem}_{day}{suffix}"
    return out_path, out_path.with_name(f"{out_path.stem}_manifest.json")
