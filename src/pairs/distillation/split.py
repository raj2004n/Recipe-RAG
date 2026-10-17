import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ...core.checksum import file_checksum
from .. import add_to_dvc, write_manifest
from .manifest import write_jsonl
from .selection import load_answers, load_categories
from .types import (
    ANSWERS_JSONL,
    CATEGORIES_JSONL,
    DEV_DRAW,
    EVAL_DRAW,
    EXPECTED_CATEGORY,
    EXPECTED_POOLS,
    EXPECTED_SPLIT,
    LEAK_RULE,
    SEED,
    dated_paths,
)

logger = logging.getLogger(__name__)


def pool_name(category: str, is_partner: bool) -> str:
    """The pool a row's primary belongs to: its category, and whether it doubles as a partner."""
    return f"{category}_{'with_partners' if is_partner else 'primary_only'}"


def build_split(rows: list[dict], categories: dict[int, str],
                rng: random.Random) -> tuple[list[str], dict[str, list[int]], dict[int, str]]:
    """Sort every answer row into train, dev or eval, and return the pools it was drawn from.

    The eval leaves first, out of the three pools whose sizes were settled, and the dev slice is
    then drawn from what remains within each category. The draw is made over row positions held in
    file order, so one seed rebuilds one split.
    """
    partners = Counter(row["partner_id"] for row in rows)
    pools: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        category = categories.get(row["primary_id"])
        if category is None:
            raise ValueError(f"the category sidecar does not cover recipe {row['primary_id']}")
        pools[pool_name(category, row["primary_id"] in partners)].append(position)

    for name, expected in EXPECTED_POOLS.items():
        if len(pools[name]) != expected:
            raise ValueError(f"pool {name} holds {len(pools[name])} rows, expected {expected}")

    labels = ["train"] * len(rows)
    for name, size in EVAL_DRAW.items():
        for position in rng.sample(pools[name], size):
            labels[position] = "eval"

    remaining: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        if label == "train":
            remaining[categories[rows[position]["primary_id"]]].append(position)
    for category, size in DEV_DRAW.items():
        for position in rng.sample(remaining[category], size):
            labels[position] = "dev"

    pool_of = {position: name for name, members in pools.items() for position in members}
    return labels, dict(pools), pool_of


def count_split(rows: list[dict], categories: dict[int, str], labels: list[str]) -> dict:
    """The split's own counts, and a refusal to write a split that is not the one settled."""
    counts = Counter(labels)
    by_category: dict[str, Counter] = defaultdict(Counter)
    for row, label in zip(rows, labels):
        by_category[label][categories[row["primary_id"]]] += 1

    for name, expected in EXPECTED_SPLIT.items():
        if counts[name] != expected:
            raise ValueError(f"{name} holds {counts[name]} rows, expected {expected}")
    for name, expected in EXPECTED_CATEGORY.items():
        if dict(by_category[name]) != expected:
            raise ValueError(f"{name} holds {dict(by_category[name])}, expected {expected}")

    return {
        "rows": len(rows),
        **{name: counts[name] for name in ("train", "dev", "eval")},
        "by_category": {name: dict(by_category[name]) for name in ("train", "dev", "eval")},
    }


def leak_report(rows: list[dict], categories: dict[int, str], labels: list[str],
                training: set[int]) -> dict:
    """What a held-out row still shares with the rows behind the given training positions.

    A partner can be trained while the primary it was paired with is held out, and a primary can be
    absent from training while the dish itself is known through some other row. Both readings are
    counted, so a reader can tell a genuinely unseen dish from a row that only looks unseen.
    """
    seen = {rows[position]["primary_id"] for position in training}
    seen |= {rows[position]["partner_id"] for position in training}
    partner_slot = {rows[position]["partner_id"] for position in training}
    evaluated = [position for position, label in enumerate(labels) if label == "eval"]
    sweet = [position for position in evaluated if categories[rows[position]["primary_id"]] == "sweet"]

    partner_trained = sum(1 for position in evaluated if rows[position]["partner_id"] in seen)
    primary_in_partner_slot = sum(1 for position in evaluated if rows[position]["primary_id"] in partner_slot)
    primary_anywhere = sum(1 for position in evaluated if rows[position]["primary_id"] in seen)

    return {
        "eval_rows": len(evaluated),
        "rows_with_a_trained_partner": partner_trained,
        "rows_whose_primary_is_absent_from_the_partner_slot": len(evaluated) - primary_in_partner_slot,
        "rows_whose_primary_appears_anywhere_in_training": primary_anywhere,
        "sweet_eval_rows": len(sweet),
        "sweet_rows_whose_primary_appears_in_training": sum(
            1 for position in sweet if rows[position]["primary_id"] in seen),
    }


def draw_split(
    answers_path: Path = ANSWERS_JSONL,
    categories_path: Path = CATEGORIES_JSONL,
    seed: int = SEED,
    output_path: Path | None = None,
    stamp: str | None = None,
) -> int:
    """Run the pass: draw the split, check its counts, write the rows and the manifest.

    The landing spot follows the dated default unless `output_path` names one, and the stamp dates
    the default in place of today. The manifest goes down before the rows and again after them, so a
    pass that dies part way through still says which split it was drawing.
    """
    answers_path, categories_path = Path(answers_path), Path(categories_path)
    out_path, manifest_path = dated_paths("distillation_split", ".jsonl", stamp)
    if output_path is not None:
        out_path = Path(output_path)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")

    rows = load_answers(answers_path)
    categories = load_categories(categories_path)
    labels, pools, pool_of = build_split(rows, categories, random.Random(seed))
    counts = count_split(rows, categories, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = {name: [i for i, label in enumerate(labels) if label == name]
                 for name in ("train", "dev", "eval")}
    pool_document = {
        name: {"rows": len(members), "recipes": len({rows[i]["primary_id"] for i in members})}
        for name, members in sorted(pools.items())
    }

    document = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": "split",
        "output": str(out_path),
        "seed": seed,
        "answers_file": str(answers_path),
        "answers_checksum": file_checksum(answers_path),
        "categories_file": str(categories_path),
        "categories_checksum": file_checksum(categories_path),
        "rule": (
            "rows are drawn, not recipes, because a recipe carries two primary rows and a recipe "
            "level hold-out can only give even counts. The eval takes 297 rows from the savoury "
            "recipes that are never a partner, 74 from the sweet recipes that are never a partner "
            "and 223 from the sweet recipes that are, which makes it 297 savoury rows against 297 "
            "sweet. The dev slice is then drawn from the rows left within each category, shaped "
            "like the training rows it comes from. Everything else trains."
        ),
        "draw": {"eval": dict(EVAL_DRAW), "dev": dict(DEV_DRAW)},
        "pools": pool_document,
        "counts": counts,
        "leak": {
            "measured_against": "the training rows the model learns from",
            "trained_rows": {
                **leak_report(rows, categories, labels, set(positions["train"])),
                "rows_behind_this_reading": len(positions["train"]),
            },
            "also_measured_for_comparison": {
                "measured_against": "everything the eval does not hold, dev included",
                **leak_report(rows, categories, labels,
                              set(positions["train"]) | set(positions["dev"])),
                "rows_behind_this_reading": len(positions["train"]) + len(positions["dev"]),
            },
            "definitions": LEAK_RULE,
        },
    }

    # The record goes down before the rows, so a pass that dies part way through still says which
    # split it was drawing.
    write_manifest(manifest_path, {**document, "state": "running"})
    write_jsonl(out_path, [
        {
            "row_id": position,
            "primary_id": row["primary_id"],
            "partner_id": row["partner_id"],
            "primary_category": categories[row["primary_id"]],
            "pool": pool_of[position],
            "split": labels[position],
        }
        for position, row in enumerate(rows)
    ])

    # The rows are whole, so the pointer moves before the record calls the pass complete.
    add_to_dvc(out_path)

    write_manifest(manifest_path, {**document, "state": "complete"})

    logger.info("wrote %d rows to %s", len(rows), out_path)
    print(f"train {counts['train']}  dev {counts['dev']}  eval {counts['eval']}  of {counts['rows']}")
    print("by category: " + "  ".join(
        f"{name} savoury={counts['by_category'][name]['savoury']} sweet={counts['by_category'][name]['sweet']}"
        for name in ("train", "dev", "eval")))
    print("pools: " + "  ".join(f"{name}={body['rows']}" for name, body in pool_document.items()))
    return 0
