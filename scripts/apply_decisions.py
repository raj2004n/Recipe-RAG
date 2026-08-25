"""CLI: read decisions journals, apply them to the candidates jsonl, write golden.jsonl.

Decisions journals are produced by the export CLI's Lavish batch-review surface
(`.decisions_batch_<n>.jsonl`). Each journal line has the shape:
  {"candidate_index": int, "decision": "correct" | "fix" | "drop",
   "corrections": {"query"?, "recipe_ids"?, "reference_answer"?, "query_type"?},
   "original": <full original row>}

This CLI merges the journal(s) with the candidates jsonl to produce the cleaned
golden.jsonl. Rules:
  - correct -> the row is copied, with verified: True.
  - fix     -> the row is copied with corrections applied, with verified: True.
  - drop    -> the row is omitted.
  - rows with no decision -> omitted (treated as not-yet-reviewed).
"""

import argparse
import json
from pathlib import Path

from src.rag.core.config import DATA_PATHS


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_recipe_ids(value: str | None) -> list[int] | None:
    """Parse a comma-separated string of recipe ids; None or empty -> None."""
    if not value:
        return None
    out: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            continue
    return out or None


def _apply_corrections(row: dict, corrections: dict) -> dict:
    """Return a new row with non-null corrections applied in place."""
    out = dict(row)
    if "query" in corrections and corrections["query"] is not None:
        out["query"] = corrections["query"]
    if "recipe_ids" in corrections and corrections["recipe_ids"] is not None:
        parsed = _parse_recipe_ids(corrections["recipe_ids"])
        if parsed is not None:
            out["recipe_ids"] = parsed
    if "reference_answer" in corrections and corrections["reference_answer"] is not None:
        out["reference_answer"] = corrections["reference_answer"]
    if "query_type" in corrections and corrections["query_type"] is not None:
        out["query_type"] = corrections["query_type"]
    out["verified"] = True
    return out


def run(args: argparse.Namespace) -> int:
    candidates_path = Path(args.candidates or DATA_PATHS["eval_golden_candidates"])
    output_path = Path(args.output or DATA_PATHS["eval_golden"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(candidates_path)
    decisions: dict[int, dict] = {}
    journals = list(args.journal) if args.journal else sorted(candidates_path.parent.glob(".decisions_batch_*.jsonl"))
    for journal in journals:
        for entry in _read_jsonl(Path(journal)):
            idx = entry.get("candidate_index")
            if isinstance(idx, int):
                decisions[idx] = entry

    kept = 0
    dropped = 0
    skipped = 0
    with output_path.open("w") as f:
        for idx, row in enumerate(rows):
            entry = decisions.get(idx)
            if entry is None:
                skipped += 1
                continue
            decision = entry.get("decision")
            if decision == "drop":
                dropped += 1
                continue
            if decision in ("correct", "fix"):
                corrections = entry.get("corrections") or {}
                final = row if decision == "correct" else _apply_corrections(row, corrections)
                final["verified"] = True
                f.write(json.dumps(final) + "\n")
                kept += 1
            else:
                skipped += 1
    print(f"Applied {len(decisions)} decisions to {len(rows)} candidates.")
    print(f"  kept:    {kept}")
    print(f"  dropped: {dropped}")
    print(f"  skipped: {skipped}")
    print(f"Wrote {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=str, default=None, help="Override the candidates jsonl path.")
    parser.add_argument("--output", type=str, default=None, help="Override the output (golden.jsonl) path.")
    parser.add_argument(
        "--journal",
        type=str,
        action="append",
        default=None,
        help="Specific decisions journal(s) to read. Defaults to all .decisions_batch_*.jsonl in the candidates dir.",
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
