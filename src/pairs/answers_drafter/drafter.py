import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..drafter_common import load_recipes
from .rendering import assert_query_types_covered, call_model
from .selection import AnswerSelection, load_prompts, selections_from_prompts


def _load_processed_keys(output_path: Path) -> set[tuple[str, int]]:
    """Resume key: (query_type, primary_id) pairs already present in the output."""
    if not output_path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with output_path.open() as f:
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


def _selection_to_row(
    selection: AnswerSelection,
    answer: str,
    model_id: str,
) -> dict[str, Any]:
    """Translate one (AnswerSelection, model output) into the schema row.

    Carries the prompt row forward and adds the answer-side fields. The
    ``context_chunks`` block is updated to include the freshly attached
    sibling chunks under the same key the prompts drafter used (so the
    verify step can render the full row in one go).
    """
    context_chunks = dict(selection.context_chunks)
    if selection.query_type == "recipe_with_a_swap":
        context_chunks.setdefault("reference", selection.reference_text or "")
        context_chunks.setdefault("swap_target", selection.swap_text or "")
    elif selection.sibling_texts:
        context_chunks.setdefault("siblings", list(selection.sibling_texts))

    return {
        "query": selection.query,
        "query_type": selection.query_type,
        "prompt_template": selection.prompt_template,
        "recipe_ids": selection.recipe_ids,
        "context_chunks": context_chunks,
        "_primary_id": selection.primary_id,
        "source_prompt": selection.source_prompt,
        "source_answer": "llm-draft-answer",
        "answer": answer,
        "answer_verified": False,
        "model_id": model_id,
    }


def draft_answers(
    prompts_path: Path,
    recipes_path: Path,
    output_path: Path,
    model_id: str,
    max_retries: int,
    backoff_base: float,
    max_workers: int,
) -> None:
    """Run the full drafter: load prompts, attach siblings, call model, write rows.

    Re-running this function resumes from ``output_path``: any (query_type,
    primary_id) pair already present (or recorded as failed in the sidecar)
    is skipped. The new file is written to a sibling temp path and then
    atomically renamed over ``output_path`` so a crashed run leaves the
    previous output in place rather than moving it aside and never replacing
    it. The failed sidecar is archived before truncation because its only
    job is to record known-bad selections for resume.
    """
    assert_query_types_covered()

    prompts = load_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"no prompt rows in {prompts_path}")

    recipes = load_recipes(recipes_path)
    if not recipes:
        raise ValueError(f"no recipes in {recipes_path}")

    selections = selections_from_prompts(prompts, recipes)

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
        for fut in tqdm(
            as_completed(future_to_sel),
            total=len(future_to_sel),
            desc="Drafting answers",
        ):
            sel = future_to_sel[fut]
            try:
                answer = fut.result()
            except Exception as e:  # noqa: BLE001
                new_failed.append((sel.query_type, sel.primary_id, str(e)))
                continue
            if not answer:
                new_failed.append((sel.query_type, sel.primary_id, "empty answer"))
                continue
            row = _selection_to_row(sel, answer, model_id)
            rows.append(row)
            processed.add((sel.query_type, sel.primary_id))

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, output_path)

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
    print(f"Drafted {len(rows)} answers; saved to {output_path}")
