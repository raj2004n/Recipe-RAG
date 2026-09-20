"""Design one writing prompt per recipe pair, for a teacher model to answer.

Each row of the pair set carries two recipe ids that a chooser decided belong together. This pass
reads both recipes and asks a model to write the single request that will be handed to a teacher
model alongside the same pair, so the teacher invents one new recipe from it. That request is the
whole of the steering the teacher gets, so it has to carry the idea the pair shares and name the
thing that must be new, without naming either dish, quoting an amount, or describing a step.

Each recipe reaches the designer as its name, its ingredients and its instructions, and nothing
else. Servings, notes and info sections say nothing about what the dish is, and the pair was
judged on how the two are cooked rather than on how the page reads. The two sections come from the
chunker's formatters, so a prompt is designed from the same text the sparse index is built out of.

The output is a dated sidecar keyed by the pair, and the pair set is never written, so a new pass
lands beside the last one rather than on top of it. Rows are appended and flushed as each call
returns, so a pass that dies keeps everything it had finished, and the file is re-sorted by rename
once the pass ends. The manifest is written before the first call and again at the end, and its
state field is what tells a finished pass from a killed one. Each row names the writer, its
sampling settings and the hash of the instructions it was given, which is what keeps an older row
attributable once the instructions move on. A row also carries the flags a read-through would
raise, a prompt that names a parent dish or quotes an amount among them, so the review list is a
filter over the file rather than a second pass over it.

A run with --limit designs a handful of pairs and writes to the smoke paths, printing every prompt
with the sections behind it, which is how the instructions get read before the corpus-wide run
pays for them.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from src.rag.core.datasets.resolver import LocalDatasetResolver
from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import registry
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.data.preprocessor import Recipe
from src.rag.retrieval.chunker import ingredient_chunks, instruction_chunks

logger = logging.getLogger(__name__)

# The pair set this reads and where the designed prompts land. A pass writes to a dated file, so
# the next pass sits beside it rather than on top of it. A limited run writes the smoke files
# instead, which are a scratch surface and are overwritten on purpose.
PAIRS_JSONL = Path("data/pairs/recipe_pairs_20260918.jsonl")
OUTPUT_DIR = Path("data/pairs")
SMOKE_JSONL = Path("data/pairs/recipe_prompts_smoke.jsonl")
SMOKE_MANIFEST = Path("data/pairs/recipe_prompts_smoke_manifest.json")

# A run over named primaries writes here. It exists so pairs whose prompt needs a second look can
# be redesigned on their own, without disturbing the smoke slice or the full set.
PROBE_JSONL = Path("data/pairs/recipe_prompts_probe.jsonl")
PROBE_MANIFEST = Path("data/pairs/recipe_prompts_probe_manifest.json")

# The design call: greedy, seeded, and thinking off. The call reads two recipes and writes one
# request, which is a composition rather than a derivation, and the measured chooser call spent
# roughly ten times the answer on reasoning this call does not need.
MODEL_ID = "qwen3.5:397b-cloud"
TEMPERATURE = 0.0
SEED = 42
THINK = False
NUM_CTX = 16384
NUM_PREDICT = 1200

# Sampling retries. A call that fails to parse or validate is retried with backoff; a pair whose
# retries all fail is reported in the manifest rather than dropped from the set.
MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_WORKERS = 3

# The version of the instructions these rows were designed under. Changing them means changing
# this, so rows written under the old instructions stay identifiable.
METHOD = "pair-prompt-v1"

# The word band a prompt is expected to land in, and the patterns a read-through would raise a
# flag about. Flags never drop a row: they sort the review list. A time is deliberately not an
# amount, because a constraint like "twenty minutes" is one of the moves the instructions invite.
WORDS_MIN = 15
WORDS_MAX = 90
QUANTITY_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:g|kg|ml|l|cups?|tbsps?|tsps?|oz|lb|grams?|°c|°f)\b",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"(?:^|\s)\d+[.)]\s")
META_PHRASES = (
    "the pair", "the two recipes above", "recipe 1", "recipe 2", "your task",
    "the rubric", "a novel recipe", "the following recipes", "as an ai",
)
# Title words that carry no dish identity, so a shared one is not a sign of copying.
TITLE_STOPWORDS = {
    "and", "with", "the", "a", "an", "of", "for", "to", "in", "on", "or", "my", "your",
    "easy", "best", "quick", "simple", "recipe", "homemade", "style", "way", "how",
}

SYSTEM_PROMPT = """You design one writing prompt from a pair of recipes.

You are given two recipes, each with its name, its ingredients and its instructions. Read both
before you write anything, and find what they share. What they share is the raw material for the
prompt you are about to write.

Your prompt is not for you. It goes to another model, which will read it alongside the same two
recipes, and that model has to invent one new recipe from it. Your prompt is the only steering
it gets, so everything that matters has to be in it.

Write a prompt that asks for a recipe neither of these two is. Not a variation, not a lighter or
faster version, not the two of them stacked together. The shared thing is the starting point,
not the destination: keep what makes it work, then move somewhere this pair does not go.

Pick the one move the pair actually supports:

- change the format, so the same flavours land as a different thing on the plate
- change the cooking method, and with it the texture
- move the cuisine or the flavour register along its own logic
- put the shared base to a different job in the meal
- keep the shared base and change what carries it
- add a constraint that forces invention: one pan, no oven, twenty minutes, a weeknight, a
  lunchbox, nothing that needs a special trip to the shop
- take the contrast that made these two interesting side by side, and make one dish out of it
  instead of two

Choose one axis, the one this pair supports. A prompt that could have been written from any
other pair is a wasted prompt. If the two recipes share only a cuisine or a single ingredient,
work from that, and if all they share is that both are food, work from the stronger of the two.

The prompt you write must:

- read like a person asking a cooking assistant, one to three sentences, ordinary language
- name the thing to invent, so the other model cannot answer by rewriting one of the two recipes
- give enough grip to be answerable: the course or the format, two to four components or
  techniques, the flavour direction, and one constraint if the pair suggests one
- stay out of the kitchen's way: no quantities, no ingredient list, no numbered steps, no method
- be one request for one recipe, not a menu, not options, not a list of ideas
- stand on its own, with nothing about how to answer it and nothing about a task, a model or
  a set of rules

You may point at the two recipes as "these two", because the other model sees them too. What you
may never do is name either dish, quote either title, name an amount, describe a step, or ask for
a mashup of two named dishes. Ask for a dish that could only have come from understanding both.

Never write the recipe yourself. You are writing the request, not the answer.

Reply with JSON only:
{"prompt": "<the prompt, one to three sentences>", "axis": "<the one move you chose, in a few words>"}

Everything inside angle brackets is a placeholder to be replaced. Return the prompt and the move,
nothing else."""


class Design(BaseModel):
    """The designer's JSON response for one pair."""

    prompt: str = Field(min_length=1, max_length=1200)
    axis: str = Field(default="", max_length=200)


def strip_markdown_fence(content: str) -> str:
    """Drop a leading and trailing ``` fence, which some models add anyway."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def prompt_hash() -> str:
    """A stable fingerprint of the instructions the run sends, for the manifest."""
    return "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def load_pairs(path: Path) -> list[dict]:
    """Every pair row, keyed once, in the order the file holds it.

    A repeated (primary, partner) pair would otherwise be designed twice and pay twice, so the
    first row wins and the repeat is reported by the manifest rather than silently dropped.
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


def dataset_key_of(pairs: list[dict]) -> str:
    """The dataset key the pairs were chosen against, which the rows all carry."""
    keys = {row.get("dataset_key") for row in pairs}
    if len(keys) != 1 or None in keys:
        raise ValueError(f"the pair rows do not agree on one dataset key: {sorted(map(str, keys))}")
    return keys.pop()


def resolve_corpus(dataset_key: str, override: str | None) -> dict:
    """The registered corpus the recipe ids resolve in, and its registry entry.

    The ids are read out of the corpus the pairs were chosen against rather than out of whatever
    file sits at the default path, so a re-registered or moved corpus is a hard stop instead of a
    quiet redesign from different recipes.
    """
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return {"path": str(path), "checksum": None, "registered_at": None, "source": "override"}

    paths = SnapshotPaths.from_paths_config({})
    entry = registry.get_dataset(paths.registry_file, dataset_key)
    if entry is None:
        raise ValueError(
            f"the registry does not know dataset key {dataset_key!r}; "
            "pass --recipes to read the corpus from a file instead"
        )
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(dataset_key)
    return {
        "path": str(resolved.path),
        "checksum": resolved.checksum,
        "registered_at": entry.get("created_at"),
        "source": "registry",
    }


def load_recipes(path: Path) -> dict[int, Recipe]:
    """The cleaned corpus, indexed by recipe id."""
    with path.open() as handle:
        return {recipe.recipe_id: recipe for recipe in (Recipe(**row) for row in json.load(handle))}


def ingredients_text(recipe: Recipe) -> str:
    """A recipe's ingredients, rendered as the sparse index renders them."""
    return ingredient_chunks([recipe])[0].text


def instructions_text(recipe: Recipe) -> str:
    """A recipe's instructions, rendered as the sparse index renders them."""
    return instruction_chunks([recipe])[0].text


def recipe_block(label: str, recipe: Recipe) -> str:
    """One recipe as the designer reads it: the name it goes by, then its two working sections.

    Servings, notes and info sections are left out. They say nothing about what the dish is, and a
    designer reading them would be reading the page rather than the cooking.
    """
    return (
        f"{label}: {recipe.recipe_name}\n\n"
        f"{ingredients_text(recipe)}\n\n{instructions_text(recipe)}"
    )


def user_message(primary: Recipe, partner: Recipe) -> str:
    """The call's user turn: both recipes, and nothing else."""
    return f"{recipe_block('RECIPE 1', primary)}\n\n{recipe_block('RECIPE 2', partner)}"


def dated_output_paths(stamp: str | None = None) -> tuple[Path, Path]:
    """This pass's prompts and manifest, dated so a later pass does not land on them."""
    day = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"recipe_prompts_{day}.jsonl"
    return out_path, out_path.with_name(f"{out_path.stem}_manifest.json")


def title_phrases(name: str) -> set[str]:
    """The two-word phrases in a dish name that identify it, stopwords stripped from the ends.

    Phrases rather than single words, because a prompt that asks for chicken is doing its job
    while a prompt that asks for "lemonade scones" has copied the dish it was designing from.
    """
    tokens = [token for token in re.findall(r"[a-z0-9]+", name.lower())]
    return {
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if left not in TITLE_STOPWORDS and right not in TITLE_STOPWORDS
    }


def flag_prompt(prompt: str, primary_name: str, partner_name: str) -> list[str]:
    """What a read-through of this prompt would raise, in the order a reviewer meets it."""
    flags = []
    words = len(prompt.split())
    if words < WORDS_MIN or words > WORDS_MAX:
        flags.append(f"length {words} words, outside the {WORDS_MIN} to {WORDS_MAX} band")

    lowered = prompt.lower()
    for name in (primary_name, partner_name):
        copied = sorted(phrase for phrase in title_phrases(name) if phrase in lowered)
        if copied:
            flags.append(f"names a source dish: {', '.join(copied)}")
    for phrase in META_PHRASES:
        if phrase in lowered:
            flags.append(f"reads as task language: {phrase}")
    if QUANTITY_RE.search(prompt):
        flags.append("quotes a quantity")
    if STEP_RE.search(prompt):
        flags.append("numbers steps")
    return flags


class DesignerClient:
    """The chat call, its retries, and the tokens it spends.

    The retry covers what the provider produces in practice: a response wrapped in a markdown
    fence, and a response that does not validate against the shape.
    """

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
            format="json",
            options=self._options(),
        )

    def design(self, message: str) -> tuple[Design | None, dict]:
        """The validated design, with the tokens and the latency it took."""
        usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}
        started = time.monotonic()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.raw_call(message)
                usage["calls"] += 1
                usage["prompt_tokens"] += response.prompt_eval_count or 0
                usage["output_tokens"] += response.eval_count or 0
                design = Design.model_validate_json(
                    strip_markdown_fence(response.message.content or "")
                )
                usage["seconds"] = round(time.monotonic() - started, 2)
                return design, usage
            except (ValidationError, ValueError, Exception) as error:  # noqa: BLE001
                last_error = error
                usage["retries"] += 1
                if attempt < self.max_retries - 1:
                    # Jitter so parallel workers do not retry in lockstep.
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("design call failed for one pair: %s", last_error)
        usage["error"] = str(last_error)
        usage["seconds"] = round(time.monotonic() - started, 2)
        return None, usage


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


def write_manifest(path: Path, document: dict) -> None:
    """Write the manifest whole, by rename, so a reader never sees it half-written."""
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temp_path, path)


def file_checksum(path: Path) -> str:
    """A file's bytes, so a manifest names the exact input a pass read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    pairs_path = Path(args.pairs)
    pairs = load_pairs(pairs_path)
    dataset_key = dataset_key_of(pairs)
    corpus = resolve_corpus(dataset_key, args.recipes)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d pairs and %d recipes from %s", len(pairs), len(recipes), corpus["path"])

    if args.limit is not None:
        pairs = sorted(pairs, key=lambda row: (row["primary_id"], row["partner_id"]))[:args.limit]
    targeted = args.primaries_file is not None
    if targeted:
        wanted = json.loads(Path(args.primaries_file).read_text())
        if isinstance(wanted, dict):
            wanted = wanted.get("primary_ids", [])
        keep = {int(item) for item in wanted}
        pairs = [row for row in pairs if row["primary_id"] in keep]
        if not pairs:
            raise ValueError(f"{args.primaries_file} names no primary in the pair set")

    smoke = args.limit is not None
    if targeted:
        out_path, manifest_path = PROBE_JSONL, PROBE_MANIFEST
    elif smoke:
        out_path, manifest_path = SMOKE_JSONL, SMOKE_MANIFEST
    else:
        out_path, manifest_path = dated_output_paths()
    if args.out is not None:
        # An explicit landing spot wins, and the manifest follows the prompts.
        out_path = Path(args.out)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[tuple[int, int], dict] = {}
    damaged = 0
    if args.resume and out_path.exists():
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
    client = DesignerClient(model_id=args.model)
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
                "workers": args.workers,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default=str(PAIRS_JSONL),
                        help="the pair set whose rows are designed from")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--model", default=MODEL_ID, help="the model that writes the prompts")
    parser.add_argument("--limit", type=int, default=None,
                        help="design only this many pairs and write to the smoke paths")
    parser.add_argument("--primaries-file", default=None,
                        help="a JSON list of primary ids to design, written to the probe paths")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="concurrent calls")
    parser.add_argument("--out", default=None, help="write the prompts here instead")
    parser.add_argument("--resume", action="store_true",
                        help="keep the rows already in the output and design only the rest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
