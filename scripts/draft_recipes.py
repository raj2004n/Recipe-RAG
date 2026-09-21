"""Answer every pair prompt with one invented recipe, to be the student's training target.

Each row of the prompt set carries two recipe ids and the request written for them. This pass hands
that request to a teacher model with both recipes rendered above it, and keeps whatever recipe comes
back. The answers are the training targets: the student is fitted to reproduce them, so a training
row is the pair's request plus the teacher's reply to it and nothing else.

The teacher is reached through the local ollama client, against a cloud model, which is the route the
prompt drafter already takes. The call differs from the drafter's in two ways, both of them
consequences of the job. The answer is prose in the shape the instructions describe rather than a
structured object, so no JSON format is requested. And a full recipe is longer than the brief that
asked for it, so the output budget is higher.

The two recipes reach the teacher through the same source block the pipeline renders at inference
time, so the text the student trains on is the text the service will hand it. The request goes in
verbatim, since it is the user turn the student is being trained to answer.

The output is a dated sidecar keyed by the pair, so a later pass lands beside the last one rather
than on top of it. Rows are appended and flushed as each call returns, so a pass that dies keeps
everything it finished, and the half-written line a killed process leaves behind is dropped rather
than counted. A pass reads back whatever the file already holds before it starts, so re-running one
continues where it stopped and never discards a finished row; --resume decides whether the prompts
already answered are skipped or answered again. The file is re-sorted by rename once the pass ends,
so a resumed run leaves one row per prompt in id order. The manifest is
written before the first call and again at the end, and its state field is what tells a finished
pass from a killed one. Each row names the writer, its sampling settings and the hash of the
instructions it was given, and carries both the request's own hash and the answer's, so an answer
can be traced back to the exact request it answers and compared against a later rerun.

A run with --limit answers a handful of prompts and writes to the smoke paths, printing each answer
with its word count and the reason the model stopped, which is how the instructions get read against
real output before the corpus-wide run pays for them.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat

from src.rag.core.logging import configure_logging
from src.rag.data.pairs import write_manifest
from src.rag.data.pairs.selection import dataset_key_of, load_recipes, resolve_corpus
from src.rag.data.preprocessor import Recipe
from src.rag.data.source_block import user_message

logger = logging.getLogger(__name__)

# The prompt set this reads and where the answers land. A pass writes to a dated file, so the next
# pass sits beside it rather than on top of it. A limited run writes the smoke files instead, which
# are a scratch surface and are overwritten on purpose.
PROMPTS_JSONL = Path("data/pairs/recipe_prompts_20260918.jsonl")
OUTPUT_DIR = Path("data/pairs")
SMOKE_JSONL = Path("data/pairs/recipe_answers_smoke.jsonl")
SMOKE_MANIFEST = Path("data/pairs/recipe_answers_smoke_manifest.json")

# The answer call: greedy, seeded, and thinking off, matching the drafter so the writer that reads
# the pair is the same writer that answers it. The output budget is higher than the drafter's
# because a whole recipe is longer than the request that asks for one.
MODEL_ID = "qwen3.5:397b-cloud"
TEMPERATURE = 0.0
SEED = 42
THINK = False
NUM_CTX = 16384
NUM_PREDICT = 1600

# Sampling retries. A call that fails or comes back empty is retried with backoff; a prompt whose
# retries all fail is reported in the manifest rather than dropped from the set.
MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_WORKERS = 3

# The version of the instructions these answers were written under. Changing them means changing
# this, so answers written under the old instructions stay identifiable.
METHOD = "pair-answer-v1"

# The length the instructions name, so a pass can count how often the teacher ran past it.
WORD_CEILING = 450

# What every answer is written against. This is the student's persona as well as the teacher's
# brief: a behaviour the student should have is stated here once and learned from the answer that
# follows it, so the two sides have to stay in step.
SYSTEM_PROMPT = """You are a recipe developer who writes for home cooks.

You will be given a request and a small set of source recipes. Invent one new recipe that answers the request. The source recipes set the flavour and the technique; adding a supporting ingredient or step that they do not contain is fine, as long as the dish still reads as a development of them. The result has to be a new dish in its own right: never a copy, a summary, or a comparison of the sources.

Write for someone cooking in an ordinary home kitchen.

- Use plain everyday words and short sentences. No chef jargon, no restaurant plating.
- Speak to the cook as "you".
- Tell them what to look for as they go: how the pan should sound, what colour to wait for, how the mixture should feel.
- Keep to equipment and ingredients a home kitchen already has.
- Keep the whole recipe under about 450 words. If you run long, cut the notes before you cut a step.

Answer in exactly this shape and in this order:

Title: the name of the dish
One or two sentences on what the dish is and why the combination works.
Serves: <number>
Cook: <number> minutes
Ingredients:
- <amount> <ingredient>, <preparation>
Instructions:
1. <step>
Notes: <optional, only when it genuinely helps the cook>

Rules for the ingredients list:

- List every ingredient the instructions use, and nothing else. If a step uses it, it is listed. If it is listed, a step uses it.
- One line per ingredient, giving the amount and any preparation, for example "1 brown onion, finely diced".
- No duplicates and no near-duplicates. If two lines could be one line, make them one line.
- Do not list water unless the amount matters.

Rules for the instructions:

- Number the steps in the order the cook performs them.
- One action per step. Give the heat, the time, or the cue that says the step is done.
- Every step has to be possible with the equipment and ingredients you listed.
- Do not restate the ingredients list inside a step.
- When the request rules something out, respect it in both the ingredients and the steps.

Never name the source recipes and never mention that you were given them."""


def load_prompts(path: Path) -> list[dict]:
    """Every prompt row, keyed once, in the order the file holds it.

    A repeated pair would otherwise be answered twice and paid for twice, so the first row wins and
    the repeat is left out.
    """
    rows, seen = [], set()
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["primary_id"], row["partner_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def system_prompt_hash() -> str:
    """A stable fingerprint of the instructions the teacher was given, for the manifest."""
    return "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def answer_hash(answer: str) -> str:
    """A stable fingerprint of one answer, so a rerun can be compared against it."""
    return "sha256:" + hashlib.sha256(answer.encode()).hexdigest()


def dated_output_paths(stamp: str | None = None) -> tuple[Path, Path]:
    """This pass's answers and manifest, dated so a later pass does not land on them."""
    day = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"recipe_answers_{day}.jsonl"
    return out_path, out_path.with_name(f"{out_path.stem}_manifest.json")


class TeacherClient:
    """The teacher, called through the local ollama client against a cloud model."""

    def __init__(self, model_id: str = MODEL_ID, temperature: float = TEMPERATURE,
                 seed: int | None = SEED, think: bool = THINK, num_ctx: int = NUM_CTX,
                 num_predict: int = NUM_PREDICT, max_retries: int = MAX_RETRIES,
                 backoff_base: float = BACKOFF_BASE):
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _options(self) -> dict:
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def raw_call(self, message: str):
        """One call, returned whole, for a probe that inspects the wire shape."""
        return chat(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            think=self.think,
            options=self._options(),
        )

    def invent(self, message: str) -> tuple[str | None, dict]:
        """The teacher's recipe, with the tokens and the latency it took.

        An answer that comes back empty counts as a failure rather than as a row, since a blank
        target teaches the student nothing and would sit in the training set looking like data.
        """
        usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}
        started = time.monotonic()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.raw_call(message)
                usage["calls"] += 1
                usage["prompt_tokens"] += response.prompt_eval_count or 0
                usage["output_tokens"] += response.eval_count or 0
                answer = (response.message.content or "").strip()
                if not answer:
                    raise ValueError("the teacher returned an empty answer")
                usage["done_reason"] = getattr(response, "done_reason", None)
                usage["seconds"] = round(time.monotonic() - started, 2)
                return answer, usage
            except Exception as error:  # noqa: BLE001
                last_error = error
                usage["retries"] += 1
                if attempt < self.max_retries - 1:
                    # Jitter so parallel workers do not retry in lockstep.
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("teacher call failed for one prompt: %s", last_error)
        usage["error"] = str(last_error)
        usage["seconds"] = round(time.monotonic() - started, 2)
        return None, usage


def answer_one(row: dict, recipes: dict[int, Recipe], client: TeacherClient,
               created_at: str) -> tuple[dict | None, dict, str | None]:
    """One prompt's answer, or the error that stopped it."""
    primary, partner = recipes[row["primary_id"]], recipes[row["partner_id"]]
    answer, usage = client.invent(user_message(primary, partner, row["prompt"]))
    if answer is None:
        return None, usage, usage.get("error", "every attempt failed")
    out = {
        "primary_id": row["primary_id"],
        "partner_id": row["partner_id"],
        "primary_name": row["primary_name"],
        "partner_name": row["partner_name"],
        "prompt": row["prompt"],
        "request_hash": row.get("prompt_hash"),
        "answer": answer,
        "answer_hash": answer_hash(answer),
        "word_count": len(answer.split()),
        "done_reason": usage.get("done_reason"),
        "method": METHOD,
        "system_prompt_hash": system_prompt_hash(),
        "writer_model": client.model_id,
        "temperature": client.temperature,
        "seed": client.seed,
        "thinking": client.think,
        "created_at": created_at,
    }
    return out, usage, None


def summarise(rows: list[dict], skipped: list[dict], prompts_in_input: int,
              pending: int | None = None, resumed: int | None = None) -> dict:
    """The counts the manifest carries and the console prints."""
    words = [row["word_count"] for row in rows]
    return {
        "prompts_in_input": prompts_in_input,
        "answers_written": len(rows),
        "prompts_resumed": resumed or 0,
        "prompts_pending_at_start": pending if pending is not None else len(rows),
        "prompts_skipped": len(skipped),
        "by_done_reason": dict(Counter(row.get("done_reason") or "unreported" for row in rows)),
        "words": {
            "min": min(words),
            "max": max(words),
            "mean": round(sum(words) / len(words), 1),
            "over_ceiling": sum(1 for count in words if count > WORD_CEILING),
        } if words else {},
    }


def run(args: argparse.Namespace) -> int:
    prompts_path = Path(args.prompts)
    prompts = load_prompts(prompts_path)
    dataset_key = dataset_key_of(prompts)
    corpus = resolve_corpus(dataset_key, args.recipes)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d prompts and %d recipes from %s", len(prompts), len(recipes), corpus["path"])

    smoke = args.limit is not None
    if smoke:
        prompts = sorted(prompts, key=lambda row: (row["primary_id"], row["partner_id"]))[:args.limit]

    if smoke:
        out_path, manifest_path = SMOKE_JSONL, SMOKE_MANIFEST
    else:
        out_path, manifest_path = dated_output_paths()
    if args.out is not None:
        # An explicit landing spot wins, and the manifest follows the answers.
        out_path = Path(args.out)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Whatever the output already holds is read back first, whether or not this pass means to skip
    # it. The file is rewritten from this mapping once the pass ends, so reading it is what keeps a
    # second pass from discarding what the first one finished.
    done: dict[tuple[int, int], dict] = {}
    damaged = 0
    if out_path.exists():
        with out_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    # A process killed mid-write leaves its last line half-written, so a line that
                    # cannot be read is the crash's own tail rather than a row worth keeping.
                    damaged += 1
                    continue
                done[(row["primary_id"], row["partner_id"])] = row
        logger.info("found %d answers already in %s%s", len(done), out_path,
                    f" ({damaged} unreadable line(s) ignored)" if damaged else "")

    # A prompt whose recipes are missing from the corpus is reported rather than answered from one
    # side of itself, which would be a recipe invented from half a pair.
    usable, skipped = [], []
    for row in prompts:
        missing = [rid for rid in (row["primary_id"], row["partner_id"]) if rid not in recipes]
        if missing:
            skipped.append({"primary_id": row["primary_id"], "partner_id": row["partner_id"],
                            "missing_recipe_ids": missing})
        else:
            usable.append(row)

    # --resume is what decides whether an answer already on disk is skipped or written again. Without
    # it every prompt is answered afresh and the newer answer wins, which costs calls but cannot lose
    # a finished row, since the row it replaces is already in hand.
    pending = [row for row in usable
               if not (args.resume and (row["primary_id"], row["partner_id"]) in done)]
    client = TeacherClient(model_id=args.model)
    resumed_count = len(usable) - len(pending)
    pending_count = len(pending)
    logger.info("answering %d of %d prompts with %s (thinking=%s, temperature=%s)",
                pending_count, len(usable), client.model_id, client.think, client.temperature)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_kind = "smoke" if smoke else "full"
    totals = Counter()
    failures: list[dict] = []

    def manifest_document(state: str, counts: dict, usage: dict, tail: list[dict]) -> dict:
        """The pass's own record. A reader tells a finished file from a killed one by its state."""
        return {
            "created_at": created_at,
            "state": state,
            "run": run_kind,
            "prompts_file": str(prompts_path),
            "output": str(out_path),
            "method": METHOD,
            "rule": "the instructions as written in the system prompt of this script",
            "dataset": {"key": dataset_key, **corpus},
            "writer": {
                "model_id": client.model_id,
                "route": "the local ollama client, against a cloud model",
                "temperature": client.temperature,
                "seed": client.seed,
                "seed_note": "passed through for reproducible reruns; the provider may ignore it",
                "thinking": client.think,
                "num_ctx": client.num_ctx,
                "num_predict": client.num_predict,
                "max_retries": client.max_retries,
                "workers": args.workers,
                "system_prompt_hash": system_prompt_hash(),
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
        summarise(list(done.values()), skipped, len(prompts), pending=pending_count,
                  resumed=resumed_count),
        {}, []))

    def work(row: dict):
        out, usage, error = answer_one(row, recipes, client, created_at)
        return row, out, usage, error

    # Rows are appended and flushed as each call returns, so the file is the pass's own progress
    # record and a killed run keeps everything it had finished. Appending rather than truncating
    # also means a stale file from an earlier killed pass is never destroyed, and since resume is
    # keyed by pair a duplicate line is harmless. The end of the pass re-sorts the file by rename.
    with out_path.open("a") as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
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
                    logger.warning("no answer for %d and %d", row["primary_id"], row["partner_id"])
                    continue
                done[(row["primary_id"], row["partner_id"])] = out
                progress.write(json.dumps(out) + "\n")
                progress.flush()
                if smoke:
                    print(f'{row["primary_id"]:>5} + {row["partner_id"]:<5} '
                          f'{out["word_count"]} words, stopped on {out["done_reason"]}')
                    print(out["answer"])
                    print()

    # The finished file holds one row per prompt in id order, whatever order the calls came back in.
    rows = [done[key] for key in sorted(done)]
    temp_path = out_path.with_name(out_path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, out_path)

    summary = summarise(rows, skipped, len(prompts), pending=pending_count, resumed=resumed_count)
    write_manifest(manifest_path, manifest_document("complete", summary, {
        "prompt_tokens": totals["prompt_tokens"],
        "output_tokens": totals["output_tokens"],
        "calls": totals["calls"],
        "retries": totals["retries"],
    }, failures))

    logger.info("wrote %d answers to %s", len(rows), out_path)
    print(f"answers: {summary['answers_written']}  skipped: {summary['prompts_skipped']}  "
          f"resumed: {resumed_count}")
    if summary["words"]:
        print(f"words: min {summary['words']['min']}  mean {summary['words']['mean']}  "
              f"max {summary['words']['max']}  over {WORD_CEILING}: {summary['words']['over_ceiling']}")
    print("stopped on: " + "  ".join(f"{reason}={count}"
                                     for reason, count in summary["by_done_reason"].items()))
    if failures:
        print(f"failures: {len(failures)} (listed in {manifest_path})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", default=str(PROMPTS_JSONL),
                        help="the prompt set whose rows are answered")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--model", default=MODEL_ID, help="the model that writes the recipes")
    parser.add_argument("--limit", type=int, default=None,
                        help="answer only this many prompts and write to the smoke paths")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent calls")
    parser.add_argument("--out", default=None, help="write the answers here instead")
    parser.add_argument("--resume", action="store_true",
                        help="skip the prompts the output already answers, instead of answering them again")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
