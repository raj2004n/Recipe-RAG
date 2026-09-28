"""Selection: read the answers, the sidecar and the split a pass works from.

A pass starts from what an earlier one wrote. The answers file's line order is the
row address everything downstream is keyed by, the category sidecar sorts each
primary into a pool, and the split names which rows train, which are held aside and
which of those went to the dev slice. A split is read back against the counts it was
drawn to, and the tokenising pass checks it against the checksum of the answers file
it was drawn from, so a rewritten answers file is a stop rather than a set of rows
pointing at pairs that have moved.

Nothing here is written and no model is called.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from ...core.checksum import file_checksum
from .types import EXPECTED_SPLIT, OUTPUT_DIR

logger = logging.getLogger(__name__)


def load_answers(path: Path) -> list[dict]:
    """The teacher's answers in file order, which is the row address everything else refers to."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_categories(path: Path) -> dict[int, str]:
    """The category sidecar, by recipe id."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["recipe_id"]: row["category"] for row in rows}


def load_split(path: Path) -> list[dict]:
    """The split, in the row order the answers file holds."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if [row["row_id"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path} does not cover every answer row exactly once")
    counts = Counter(row["split"] for row in rows)
    if {name: counts[name] for name in EXPECTED_SPLIT} != EXPECTED_SPLIT:
        raise ValueError(f"{path} holds {dict(counts)}, not the settled split")
    return rows


def newest_split() -> Path:
    """The split a tokenising pass reads when it is not told which one."""
    candidates = sorted(OUTPUT_DIR.glob("distillation_split_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no split in {OUTPUT_DIR}; run --run split first")
    return candidates[-1]


def check_provenance(split_path: Path, answers_path: Path) -> None:
    """Refuse to tokenise a split whose answers file has moved since the split was drawn.

    A split row addresses the answers file by line, so an answers file that was rewritten would
    point every held-out row at a different pair. The manifest's checksum catches that before a row
    is rendered; the pair named on each row is the check behind it.
    """
    manifest_path = split_path.with_name(f"{split_path.stem}_manifest.json")
    if not manifest_path.exists():
        logger.warning("%s carries no manifest, so its inputs cannot be checked", split_path)
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("answers_checksum") != file_checksum(answers_path):
        raise ValueError(f"{answers_path} is not the file {split_path} was drawn from")
