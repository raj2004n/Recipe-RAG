import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ..preprocessor import Recipe
from .manifest import add_to_dvc, write_manifest
from .rendering import DesignerClient, flag_prompt, prompt_hash, user_message
from .selection import dataset_key_of, load_pairs, load_recipes, resolve_corpus
from .types import (
    MAX_WORKERS,
    METHOD,
    MODEL_ID,
    PROBE_JSONL,
    PROBE_MANIFEST,
    SMOKE_JSONL,
    SMOKE_MANIFEST,
    dated_output_paths,
)

logger = logging.getLogger(__name__)


def design_one(row: dict, recipes: dict[int, Recipe], client: DesignerClient,
               created_at: str) -> tuple[dict | None, dict, str | None]:
    """One pair's row, or the error that stopped it."""
    primary, partner = recipes[row["primary_id"]], recipes[row["partner_id"]]
    design, usage = client.design(user_message(primary, partner))
    if design is None:
        return None, usage, usage.get("error", "every attempt failed")
    prompt = design.prompt.strip()
    out = {
        "primary_id": row["primary_id"],
        "partner_id": row["partner_id"],
        "primary_name": row["primary_name"],
        "partner_name": row["partner_name"],
        "prompt": prompt,
        "axis": design.axis.strip(),
        "flags": flag_prompt(prompt, row["primary_name"], row["partner_name"]),
        "pair_index": row.get("pair_index"),
        "pair_index_of": row.get("pair_index_of"),
        "dataset_key": row.get("dataset_key"),
        "method": METHOD,
        "writer_model": client.model_id,
        "temperature": client.temperature,
        "seed": client.seed,
        "thinking": client.think,
        "prompt_hash": prompt_hash(),
        "created_at": created_at,
    }
    return out, usage, None


def summarise(rows: list[dict], skipped: list[dict], pairs_in_input: int,
              pending: int = 0, resumed: int = 0) -> dict:
    """The counts a reader needs to see the shape of the set, and the tail.

    `pending` is how many pairs the pass still had to design when it started and `resumed` how
    many it inherited from an earlier pass, which is what makes a partly finished file readable as
    a partly finished file rather than as a small corpus.
    """
    flags = Counter(flag.split(":")[0] for row in rows for flag in row["flags"])
    return {
        "pairs_in_input": pairs_in_input,
        "prompts_written": len(rows),
        "pairs_resumed": resumed,
        "pairs_pending_at_start": pending,
        "pairs_skipped": len(skipped),
        "flagged": sum(1 for row in rows if row["flags"]),
        "flagged_reasons": dict(flags),
        "by_axis": dict(Counter(row["axis"] or "unstated" for row in rows).most_common()),
    }


def design_prompts(
    pairs_path: Path,
    recipes_override: str | None = None,
    model_id: str = MODEL_ID,
    limit: int | None = None,
    primaries_file: str | None = None,
    output_path: Path | None = None,
    workers: int = MAX_WORKERS,
    resume: bool = False,
) -> int:
    """Run the pass: read the pairs, design one prompt each, write the rows.

    The landing spot follows the run kind unless `output_path` names one: a limited run writes the
    smoke files, a run over named primaries writes the probe files, and a full run writes a dated
    pair of paths so it never lands on the pass it replaces. A resumed run keeps the rows already
    in the output and designs only the pairs missing from it.
    """
    pairs = load_pairs(pairs_path)
    dataset_key = dataset_key_of(pairs)
    corpus = resolve_corpus(dataset_key, recipes_override)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d pairs and %d recipes from %s", len(pairs), len(recipes), corpus["path"])

    if limit is not None:
        pairs = sorted(pairs, key=lambda row: (row["primary_id"], row["partner_id"]))[:limit]
    targeted = primaries_file is not None
    if targeted:
        wanted = json.loads(Path(primaries_file).read_text())
        if isinstance(wanted, dict):
            wanted = wanted.get("primary_ids", [])
        keep = {int(item) for item in wanted}
        pairs = [row for row in pairs if row["primary_id"] in keep]
        if not pairs:
            raise ValueError(f"{primaries_file} names no primary in the pair set")

    smoke = limit is not None
    if targeted:
        out_path, manifest_path = PROBE_JSONL, PROBE_MANIFEST
    elif smoke:
        out_path, manifest_path = SMOKE_JSONL, SMOKE_MANIFEST
    else:
        out_path, manifest_path = dated_output_paths()
    if output_path is not None:
        # An explicit landing spot wins, and the manifest follows the prompts.
        out_path = Path(output_path)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[tuple[int, int], dict] = {}
    damaged = 0
    if resume and out_path.exists():
        with out_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    # A process killed mid-write leaves its last line half-written, so a line that
                    # cannot be read is the crash's own tail rather than a row worth reporting.
                    damaged += 1
                    continue
                done[(row["primary_id"], row["partner_id"])] = row
        logger.info("resuming from %s with %d prompts already written%s", out_path, len(done),
                    f" ({damaged} unreadable line(s) ignored)" if damaged else "")

    # A pair whose recipes are missing from the corpus is reported rather than designed from one
    # side of itself, which would be a prompt built from half a pair.
    usable, skipped = [], []
    for row in pairs:
        missing = [rid for rid in (row["primary_id"], row["partner_id"]) if rid not in recipes]
        if missing:
            skipped.append({"primary_id": row["primary_id"], "partner_id": row["partner_id"],
                            "missing_recipe_ids": missing})
        else:
            usable.append(row)

    pending = [row for row in usable if (row["primary_id"], row["partner_id"]) not in done]
    client = DesignerClient(model_id=model_id)
    resumed_count, pending_count = len(done), len(pending)
    logger.info("designing %d of %d prompts with %s (thinking=%s, temperature=%s)",
                pending_count, len(usable), client.model_id, client.think, client.temperature)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_kind = "probe" if targeted else ("smoke" if smoke else "full")
    totals = Counter()
    failures: list[dict] = []

    def manifest_document(state: str, counts: dict, usage: dict, tail: list[dict]) -> dict:
        """The pass's own record. A reader tells a finished file from a killed one by its state."""
        return {
            "created_at": created_at,
            "state": state,
            "run": run_kind,
            "pairs_file": str(pairs_path),
            "output": str(out_path),
            "method": METHOD,
            "rule": "the instructions as written in the system prompt of this script",
            "dataset": {"key": dataset_key, **corpus},
            "writer": {
                "model_id": client.model_id,
                "temperature": client.temperature,
                "seed": client.seed,
                "seed_note": "passed through for reproducible reruns; the provider may ignore it",
                "thinking": client.think,
                "num_ctx": client.num_ctx,
                "num_predict": client.num_predict,
                "max_retries": client.max_retries,
                "workers": workers,
                "prompt_hash": prompt_hash(),
            },
            "counts": counts,
            "usage": usage,
            "skipped": skipped,
            "failures": tail,
        }

    # The manifest goes down before the first call, so a pass that never comes back still leaves a
    # dated record of what it was doing, how much it had inherited and how much it still owed.
    write_manifest(manifest_path, manifest_document(
        "running",
        summarise(list(done.values()), skipped, len(pairs), pending=pending_count,
                  resumed=resumed_count),
        {}, []))

    def work(row: dict):
        out, usage, error = design_one(row, recipes, client, created_at)
        return row, out, usage, error

    # Rows are appended and flushed as each call returns, so the file is the pass's own progress
    # record and a killed run keeps everything it had finished. Appending rather than truncating
    # also means a stale file from an earlier killed pass is never destroyed, and since resume is
    # keyed by pair a duplicate line is harmless. The end of the pass re-sorts the file by rename.
    with out_path.open("a") as progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, row) for row in pending]
            for future in as_completed(futures):
                row, out, usage, error = future.result()
                totals["calls"] += usage["calls"]
                totals["retries"] += usage["retries"]
                totals["prompt_tokens"] += usage["prompt_tokens"]
                totals["output_tokens"] += usage["output_tokens"]
                if out is None:
                    failures.append({"primary_id": row["primary_id"],
                                     "partner_id": row["partner_id"], "error": error})
                    logger.warning("no prompt for %d and %d", row["primary_id"], row["partner_id"])
                    continue
                done[(row["primary_id"], row["partner_id"])] = out
                progress.write(json.dumps(out) + "\n")
                progress.flush()
                if smoke or targeted:
                    print(f'{row["primary_id"]:>5} + {row["partner_id"]:<5} {out["axis"]}')
                    print(f'      {out["prompt"]}')
                    for flag in out["flags"]:
                        print(f'      flag: {flag}')
                    print()

    # The finished file holds one row per pair in id order, whatever order the calls came back in.
    rows = [done[key] for key in sorted(done)]
    temp_path = out_path.with_name(out_path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, out_path)

    # The full run is the set the passes downstream read, so it is the one that gets a
    # pointer. The smoke and probe files are scratch, and are left untracked.
    if run_kind == "full":
        add_to_dvc(out_path)

    summary = summarise(rows, skipped, len(pairs), pending=pending_count, resumed=resumed_count)
    write_manifest(manifest_path, manifest_document("complete", summary, {
        "prompt_tokens": totals["prompt_tokens"],
        "output_tokens": totals["output_tokens"],
        "calls": totals["calls"],
        "retries": totals["retries"],
    }, failures))

    logger.info("wrote %d prompts to %s", len(rows), out_path)
    print(f"prompts: {summary['prompts_written']}  flagged: {summary['flagged']}  "
          f"skipped: {summary['pairs_skipped']}  resumed: {resumed_count}")
    print("axes: " + "  ".join(f"{axis}={count}" for axis, count in summary["by_axis"].items()))
    if failures:
        print(f"failures: {len(failures)} (listed in {manifest_path})")
    return 0
