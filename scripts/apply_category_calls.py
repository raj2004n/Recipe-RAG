"""Apply the reviewed category calls to the recipe label sidecar.

The labelling pass left fourteen titles unresolved rather than guessing at them,
and a review settled every one by hand: six to sweet, two to savoury, and six ids
that are not dishes at all, which left the pair set instead. The removals were
scoped to the pair file. The eight relabelled rows are changes to the sidecar,
and this is the step that writes them.

A human call is not the same kind of evidence as a model call, so it does not
overwrite it. The category field carries the settled value, which is what a
reader of the sidecar needs, and the model's own answer stays in the row under
human_call, so the row keeps its history and the two kinds of call stay
distinguishable. Confidence is left as the model wrote it, because it is a
reading of how much the title settled on its own, not a mark of who called it.

The manifest gains a human_calls block naming every row that moved and the
counts the file holds after the write, so the labelling run's own numbers stay
as the record of that run while the current state of the sidecar is legible from
the same place. Lines that do not move are copied through untouched, which keeps
the diff to exactly the rows that were called.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SIDECAR_JSONL = Path("data/pairs/recipe_categories.jsonl")
SIDECAR_MANIFEST = Path("data/pairs/recipe_categories_manifest.json")

# The review's calls, six rows to sweet and two to savoury. The six ids that
# left the pair set are absent here: they are not sidecar rows.
CALLS: dict[int, str] = {
    53: "sweet",
    154: "savoury",
    238: "savoury",
    337: "sweet",
    407: "sweet",
    424: "sweet",
    1186: "sweet",
    1274: "sweet",
}

# Every one of these rows was called because the pass had abstained on it, so
# the write expects to find the row still unresolved. A sidecar that reads
# otherwise has moved on since the review, and the run stops rather than
# guessing at which answer is current.
EXPECTED_FROM = "unresolved"

SOURCE = "category review, batch of fourteen calls, September 2026"


def _read_rows(raw: list[str]) -> list[dict]:
    return [json.loads(line) for line in raw]


def _called_row(row: dict, category: str, decided_at: str) -> dict:
    """Return the row with the reviewed call settled and the model's kept."""
    out = dict(row)
    out["category"] = category
    out["human_call"] = {
        "category": category,
        "previous_category": row["category"],
        "decided_at": decided_at,
        "source": SOURCE,
    }
    return out


def run(args: argparse.Namespace) -> int:
    sidecar_path = Path(args.sidecar)
    manifest_path = Path(args.manifest)

    raw = sidecar_path.read_text().splitlines(keepends=True)
    original_raw = list(raw)
    rows = _read_rows(raw)
    index = {row["recipe_id"]: position for position, row in enumerate(rows)}

    missing = sorted(recipe_id for recipe_id in CALLS if recipe_id not in index)
    if missing:
        raise SystemExit(f"recipe ids absent from {sidecar_path}: {missing}")
    stale = sorted(
        recipe_id for recipe_id in CALLS
        if rows[index[recipe_id]]["category"] != EXPECTED_FROM
    )
    if stale:
        seen = {recipe_id: rows[index[recipe_id]]["category"] for recipe_id in stale}
        raise SystemExit(
            f"expected {EXPECTED_FROM} on every called row, found {seen}. "
            "The sidecar has changed since the review; nothing was written."
        )

    before = Counter(row["category"] for row in rows)
    decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    moved: list[dict] = []
    for recipe_id, category in sorted(CALLS.items()):
        position = index[recipe_id]
        row = rows[position]
        rows[position] = _called_row(row, category, decided_at)
        raw[position] = json.dumps(rows[position]) + "\n"
        moved.append({
            "recipe_id": recipe_id,
            "name": row["name"],
            "from": row["category"],
            "to": category,
        })
        print(f"{recipe_id:>5}  {row['category']:>10} -> {category:<8}  {row['name']}")

    after = Counter(row["category"] for row in rows)
    print("counts: " + "  ".join(
        f"{name}={before.get(name, 0)}->{after.get(name, 0)}"
        for name in sorted(set(before) | set(after))))
    print(f"lines changed: {sum(1 for was, now in zip(original_raw, raw) if was != now)}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    temporary = sidecar_path.with_name(sidecar_path.name + ".tmp")
    temporary.write_text("".join(raw))
    temporary.replace(sidecar_path)

    manifest = json.loads(manifest_path.read_text())
    manifest["updated_at"] = decided_at
    manifest["human_calls"] = {
        "applied_at": decided_at,
        "source": SOURCE,
        "calls": moved,
        "counts_after": dict(sorted(after.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {sidecar_path} and {manifest_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", default=str(SIDECAR_JSONL),
                        help="the label sidecar to write the calls into")
    parser.add_argument("--manifest", default=str(SIDECAR_MANIFEST),
                        help="the manifest to record the calls in")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the calls and the resulting counts, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
