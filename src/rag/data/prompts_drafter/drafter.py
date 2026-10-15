"""Orchestration: run selection, model calls, and atomic output for one full draft."""

import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ...retrieval.dense.encoder import Encoder
from ..drafter_common import load_recipes
from ..pairs import add_to_dvc
from .rendering import call_model, prompt_template_tag
from .selection import Selection, selections_for_type
from .types import QUERY_TYPES, TYPES_WITH_ONE_PRIMARY


def _load_existing_rows(output_path: Path) -> list[dict[str, Any]]:
    """Read the rows already on disk so a resume run can merge them back."""
    if not output_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_processed_keys(output_path: Path) -> set[tuple[str, int]]:
    """Resume key: (query_type, primary_id) pairs already present in the output."""
    keys: set[tuple[str, int]] = set()
    for row in _load_existing_rows(output_path):
        try:
            keys.add((row["query_type"], int(row["_primary_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def _failed_path(output_path: Path) -> Path:
    """Sidecar for selections the model couldn't draft. Resume skips them."""
    return output_path.with_name(f"{output_path.name}.failed.jsonl")


def _load_failed_keys(output_path: Path) -> set[tuple[str, int]]:
    """Read (query_type, primary_id) pairs that already failed a previous run."""
    p = _failed_path(output_path)
    if not p.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add((row["query_type"], int(row["_primary_id"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def _archive_failed(output_path: Path) -> None:
    """Move an existing failed sidecar to a timestamped backup before truncation."""
    failed_path = _failed_path(output_path)
    if not failed_path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = failed_path.with_name(f"{failed_path.name}.bak.{timestamp}")
    shutil.move(str(failed_path), str(backup))


def _context_chunks_for_selection(selection: Selection) -> dict[str, Any]:
    """Build the ``context_chunks`` block stored on each row."""
    if selection.query_type == "info":
        return {"info_chunk": selection.primary_text}
    if selection.query_type == "recipe_with_a_swap":
        return {
            "primary": selection.primary_text,
            "swap_targets": [selection.swap_text],
        }
    if selection.query_type in TYPES_WITH_ONE_PRIMARY:
        return {"primary": selection.primary_text}
    # recipe_suggestion_off_* types.
    return {
        "primary": selection.primary_text,
        "siblings": selection.sibling_texts,
    }


def _selection_to_row(
    selection: Selection,
    query: str,
) -> dict[str, Any]:
    """Translate one (selection, model output) into the schema row."""
    recipe_ids = [selection.primary_recipe_id, *selection.sibling_recipe_ids]
    context_chunks = _context_chunks_for_selection(selection)
    return {
        "query": query,
        "recipe_ids": recipe_ids,
        "context_chunks": context_chunks,
        "query_type": selection.query_type,
        "prompt_template": prompt_template_tag(selection.query_type),
        "source": "llm-draft-prompt",
        "prompt_verified": False,
        # Internal field used by resume; downstream tooling can ignore.
        "_primary_id": selection.primary_id,
    }


def draft_prompts(
    recipes_path: Path,
    output_path: Path,
    encoder: Encoder,
    model_id: str,
    max_retries: int,
    backoff_base: float,
    max_workers: int,
    per_type_count: int,
    seed: int | None = None,
) -> None:
    """Run the full drafter: select, call the model, write rows.

    Re-running this function resumes from ``output_path``: any (query_type,
    primary_id) pair already present (or recorded as failed in the sidecar)
    is skipped, and the rows already on disk are preserved and merged with
    the newly drafted ones. The merged file is written to a sibling temp
    path and then atomically renamed over ``output_path`` so a crashed run
    leaves the existing file in place rather than moving it aside and never
    replacing it. Once the replacement is in place the run records its own DVC
    pointer, since the file is whole at that point and the rows in it are what
    the pointer stands for. The failed sidecar is archived before truncation
    because its only job is to record known-bad selections for resume.
    """
    recipes = load_recipes(recipes_path)
    if not recipes:
        raise ValueError(f"no recipes in {recipes_path}")

    rng = random.Random(seed)

    selections: list[Selection] = []
    for query_type in QUERY_TYPES:
        selections.extend(selections_for_type(
            query_type=query_type,
            per_type_count=per_type_count,
            recipes=recipes,
            encoder=encoder,
            rng=rng,
        ))

    existing_rows = _load_existing_rows(output_path)
    processed = _load_processed_keys(output_path)
    failed = _load_failed_keys(output_path)
    _archive_failed(output_path)

    rows: list[dict[str, Any]] = []
    new_failed: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sel = {
            executor.submit(
                call_model,
                sel,
                model_id,
                max_retries,
                backoff_base,
            ): sel
            for sel in selections
            if (sel.query_type, sel.primary_id) not in processed
            and (sel.query_type, sel.primary_id) not in failed
        }
        for fut in tqdm(as_completed(future_to_sel), total=len(future_to_sel), desc="Drafting prompts"):
            sel = future_to_sel[fut]
            try:
                query = fut.result()
            except Exception as e:  # noqa: BLE001
                new_failed.append((sel.query_type, sel.primary_id, str(e)))
                continue
            if not query:
                new_failed.append((sel.query_type, sel.primary_id, "empty query"))
                continue
            row = _selection_to_row(sel, query)
            rows.append(row)
            processed.add((sel.query_type, sel.primary_id))

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in existing_rows:
            f.write(json.dumps(row) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, output_path)
    add_to_dvc(output_path)

    failed_path = _failed_path(output_path)
    if new_failed:
        with failed_path.open("w") as f:
            for qt, pid, err in new_failed:
                f.write(json.dumps({
                    "query_type": qt,
                    "_primary_id": pid,
                    "error": err,
                }) + "\n")
        preview = [(qt, pid) for qt, pid, _ in new_failed[:10]]
        print(f"Failed to draft {len(new_failed)} selections: {preview}")
        print(f"Recorded failures to {failed_path}; rerun will skip these.")
    print(f"Drafted {len(rows)} prompts; saved to {output_path}")
