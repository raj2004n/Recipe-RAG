"""Draft additional user-question prompts and append them to ``prompts.jsonl``.

The drafter in ``scripts.draft_prompts`` builds a fresh
``prompts.jsonl`` from scratch, archiving any existing file aside before
writing. This script is the other half of the workflow: when a few extra
rows are needed for one or two query types, it drafts them with the same
selection and model-call helpers as the main drafter and appends them to
the existing file, leaving the rows already on disk untouched.

The collision contract is the same as the main drafter's resume rule:
any ``(query_type, _primary_id)`` pair already present on disk is
skipped, so re-running the script with the same seed is a no-op once the
requested rows have landed.
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.rag.data.drafter_common import build_encoder, draft_kwargs, resolve_paths
from src.rag.data.prompts_drafter import QUERY_TYPES
from src.rag.data.prompts_drafter.drafter import _selection_to_row
from src.rag.data.prompts_drafter.rendering import call_model
from src.rag.data.prompts_drafter.selection import selections_for_type


def _parse_types(spec: str) -> dict[str, int]:
    """Parse a ``type:count,type:count`` string into a dict."""
    out: dict[str, int] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"--types entry must be 'type:count', got {piece!r}")
        key, _, count_str = piece.partition(":")
        key = key.strip()
        count = int(count_str.strip())
        if key not in QUERY_TYPES:
            raise ValueError(f"unknown query_type {key!r}; valid: {QUERY_TYPES}")
        out[key] = count
    if not out:
        raise ValueError("--types produced an empty spec")
    return out


def _load_existing_primary_ids(prompts_path: Path) -> dict[str, set[int]]:
    """Group existing _primary_id values by query_type, for collision avoidance."""
    by_type: dict[str, set[int]] = defaultdict(set)
    if not prompts_path.exists():
        return by_type
    with prompts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qt = row.get("query_type")
            pid = row.get("_primary_id")
            if qt is None or pid is None:
                continue
            by_type[qt].add(int(pid))
    return by_type


def run(args: argparse.Namespace) -> int:
    """Draft the requested rows and append them to ``prompts.jsonl``.

    For each ``type:count`` pair in ``--types``, the script asks the drafter's
    selection logic for several times the wanted count, drops any candidates
    whose primary id is already on disk, and keeps the first survivors. The
    model is called for those survivors in parallel and the resulting rows
    are appended to the output file.
    """
    corpus_path, prompts_path = resolve_paths(args.output)
    type_counts = _parse_types(args.types)

    from src.rag.data.drafter_common import load_recipes
    recipes = load_recipes(corpus_path)
    if not recipes:
        raise ValueError(f"no recipes in {corpus_path}")

    existing = _load_existing_primary_ids(prompts_path)
    rng = random.Random(args.seed)
    encoder = build_encoder()
    call_kwargs = draft_kwargs()
    # A small number of rows never needs the full worker pool.
    call_kwargs["max_workers"] = min(call_kwargs["max_workers"], len(type_counts) * 4 or 1)

    # Collect enough candidates per type so we can drop collisions and still hit the target.
    new_rows: list[dict] = []
    futures: dict = {}
    with ThreadPoolExecutor(max_workers=call_kwargs["max_workers"]) as executor:
        for query_type, wanted in type_counts.items():
            already_used = existing.get(query_type, set())
            # Ask for several times the wanted count so collision filtering has room.
            oversample = max(wanted * 6, 12)
            selections = selections_for_type(
                query_type=query_type,
                per_type_count=oversample,
                recipes=recipes,
                encoder=encoder,
                rng=rng,
            )
            fresh = [s for s in selections if s.primary_id not in already_used]
            if len(fresh) < wanted:
                raise RuntimeError(
                    f"only {len(fresh)} fresh {query_type} primaries available after "
                    f"collision filtering; wanted {wanted}. Re-run with --seed <other>."
                )
            for sel in fresh[:wanted]:
                futures[executor.submit(
                    call_model,
                    sel,
                    call_kwargs["model_id"],
                    call_kwargs["max_retries"],
                    call_kwargs["backoff_base"],
                )] = sel

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Backfilling prompts"):
            sel = futures[fut]
            try:
                query = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {sel.query_type} primary_id={sel.primary_id}: {e}")
                continue
            if not query:
                print(f"  EMPTY  {sel.query_type} primary_id={sel.primary_id}")
                continue
            new_rows.append(_selection_to_row(sel, query))

    if not new_rows:
        print("No new rows drafted; prompts.jsonl untouched.")
        return 0

    # Append rather than overwrite so the rows already on disk stay in place.
    with prompts_path.open("a") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Appended {len(new_rows)} backfill rows to {prompts_path}")
    for row in new_rows:
        print(f"  + {row['query_type']} primary_id={row['_primary_id']}: {row['query']!r}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--types",
        type=str,
        required=True,
        help="Comma-separated 'type:count' pairs, e.g. 'technique:2,recipe_suggestion_off_info:1'.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for the selection RNG.")
    parser.add_argument("--output", type=str, default=None, help="Override the prompts jsonl path.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
