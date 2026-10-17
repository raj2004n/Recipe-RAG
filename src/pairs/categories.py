import hashlib
import json
import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from .manifest import add_to_dvc
from .rendering import strip_markdown_fence

logger = logging.getLogger(__name__)

# The pair set this reads and where the labels land. A limited run writes the
# smoke files instead, so reading a changed prompt never lands on the real set.
INPUT_JSONL = Path("data/pairs/recipe_pairs_20260918.jsonl")
OUTPUT_JSONL = Path("data/pairs/recipe_categories.jsonl")
OUTPUT_MANIFEST = Path("data/pairs/recipe_categories_manifest.json")
SMOKE_JSONL = Path("data/pairs/recipe_categories_smoke.jsonl")
SMOKE_MANIFEST = Path("data/pairs/recipe_categories_smoke_manifest.json")

# A run over a named set of ids writes here. It exists so the titles the rule is
# least likely to settle can be read on their own, without disturbing the smoke
# slice or the full set.
PROBE_JSONL = Path("data/pairs/recipe_categories_probe.jsonl")
PROBE_MANIFEST = Path("data/pairs/recipe_categories_probe_manifest.json")

# The labelling call: greedy, seeded, and thinking off, matching how the pairs
# themselves were chosen. The judgement reads a title against a rule, which is a
# recognition rather than a derivation, and the measured chooser call spent
# roughly ten times the answer on reasoning this call does not need.
MODEL_ID = "minimax-m3:cloud"
TEMPERATURE = 0.0
SEED = 42
THINK = False
NUM_CTX = 8192
NUM_PREDICT = 400

# Sampling retries. A call that fails to parse or validate is retried with
# backoff; a title whose retries all fail is reported in the manifest rather
# than dropped from the set.
MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_WORKERS = 3

# The rule version these rows were produced under. Changing the boundary means
# changing this, so rows written under the old rule stay identifiable.
METHOD = "name-judgement-v2"

SYSTEM_PROMPT = """You label a recipe as savoury or sweet from its title alone.

Sweet means the thing being served is sweet: cakes, cookies, puddings, ice cream, confectionery, sweet drinks, sweet condiments, and sweet component recipes where the sweetness is the point.

Savoury means the dish is savoury, even when the title carries a sweet word or the recipe contains a sweet ingredient whose job is seasoning rather than the dish itself. "Sweet corn soup" is savoury. "Halloumi with honey" is savoury. "Chocolate ganache" is sweet.

A sweet drink is sweet. A cocktail, a lemonade, a liqueur or a sweet tea counts as sweet even though it is drunk rather than eaten, and a drink is never savoury for being alcoholic.

Unresolved is for a title that genuinely allows both and says nothing to choose between them. It is not for a title that needs a moment's thought: use ordinary knowledge of what the dish is before reaching for it.

Confidence is how much the title alone supports the call. High when the title names the dish outright, medium when it takes knowing what the dish is, low when the call is a judgement the title does not really settle.

Give one short clause of evidence naming what in the title decided it.

Reply with JSON only:
{"category": "savoury" | "sweet" | "unresolved", "confidence": "high" | "medium" | "low", "evidence": "<one short clause>"}

Everything inside angle brackets is a placeholder to be replaced. Return one of the three words for the category, one of the three for the confidence, and a real clause for the evidence."""


class Label(BaseModel):
    """The labeller's JSON response for one title."""

    category: Literal["savoury", "sweet", "unresolved"]
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(min_length=1, max_length=300)


def load_titles(path: Path) -> tuple[dict[int, str], list[dict]]:
    """Every recipe id the pair set names, with its title.

    Both sides of a pair are read, because a recipe that is only ever a partner
    still needs a label. A title that disagrees with itself across rows is kept
    as the first spelling and reported, since a rename is worth seeing rather
    than silently picking one.
    """
    titles: dict[int, str] = {}
    conflicts: list[dict] = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        for id_key, name_key in (("primary_id", "primary_name"), ("partner_id", "partner_name")):
            recipe_id, name = row[id_key], row[name_key]
            known = titles.get(recipe_id)
            if known is None:
                titles[recipe_id] = name
            elif known != name:
                conflicts.append({"recipe_id": recipe_id, "kept": known, "seen": name})
    return titles, conflicts


def prompt_hash() -> str:
    """A stable fingerprint of the prompt the run sends, for the manifest."""
    return "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


class LabellerClient:
    """The chat call, its retries, and the tokens it spends.

    The retry covers what the provider produces in practice: a response wrapped
    in a markdown fence, and a response that does not validate against the
    shape.
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

    def raw_call(self, title: str):
        """One call, returned whole, for a probe that inspects the wire shape."""
        return chat(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Recipe title: {title}"},
            ],
            think=self.think,
            format="json",
            options=self._options(),
        )

    def label(self, title: str) -> tuple[Label | None, dict]:
        """The validated label, with the tokens and the latency it took."""
        usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}
        started = time.monotonic()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.raw_call(title)
                usage["calls"] += 1
                usage["prompt_tokens"] += response.prompt_eval_count or 0
                usage["output_tokens"] += response.eval_count or 0
                label = Label.model_validate_json(strip_markdown_fence(response.message.content or ""))
                usage["seconds"] = round(time.monotonic() - started, 2)
                return label, usage
            except (ValidationError, ValueError, Exception) as error:  # noqa: BLE001
                last_error = error
                usage["retries"] += 1
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("labelling call failed for %r: %s", title, last_error)
        usage["error"] = str(last_error)
        usage["seconds"] = round(time.monotonic() - started, 2)
        return None, usage


def label_one(recipe_id: int, title: str, client: LabellerClient,
              created_at: str) -> tuple[dict | None, dict, str | None]:
    """One recipe's row, or the error that stopped it."""
    label, usage = client.label(title)
    if label is None:
        return None, usage, usage.get("error", "every attempt failed")
    row = {
        "recipe_id": recipe_id,
        "name": title,
        "category": label.category,
        "confidence": label.confidence,
        "evidence": label.evidence.strip(),
        "method": METHOD,
        "labeller_model": client.model_id,
        "temperature": client.temperature,
        "seed": client.seed,
        "thinking": client.think,
        "prompt_hash": prompt_hash(),
        "created_at": created_at,
    }
    return row, usage, None


def summarise(rows: list[dict]) -> dict:
    """The counts a reader needs to see the shape of the labels, and the tail.

    A row is flagged for review when the label is unresolved or when the
    confidence is anything short of high. Medium is included deliberately: the
    probe showed a model rating its own doubtful calls medium and its outright
    errors medium too, so a review list built on unresolved alone would collect
    the honest abstentions and miss the wrong answers.
    """
    by_category = Counter(row["category"] for row in rows)
    by_confidence = Counter(row["confidence"] for row in rows)
    confusion = {
        category: dict(Counter(row["confidence"] for row in rows if row["category"] == category))
        for category in ("savoury", "sweet", "unresolved")
    }
    flagged = [
        {"recipe_id": row["recipe_id"], "name": row["name"], "category": row["category"],
         "confidence": row["confidence"], "evidence": row["evidence"]}
        for row in rows
        if row["category"] == "unresolved" or row["confidence"] != "high"
    ]
    reasons = Counter(
        "unresolved" if row["category"] == "unresolved" else f"{row['confidence']} confidence"
        for row in rows
        if row["category"] == "unresolved" or row["confidence"] != "high"
    )
    return {
        "labels": len(rows),
        "by_category": dict(by_category),
        "by_confidence": dict(by_confidence),
        "confidence_by_category": confusion,
        "flagged": len(flagged),
        "flagged_reasons": dict(reasons),
        "flagged_ids": sorted(row["recipe_id"] for row in flagged),
    }


def label_categories(
    input_path: Path = INPUT_JSONL,
    model_id: str = MODEL_ID,
    limit: int | None = None,
    ids_file: str | None = None,
    output_path: Path | None = None,
    workers: int = MAX_WORKERS,
    resume: bool = False,
) -> int:
    """Run the pass: read the pair set's titles, label each one, write the rows.

    The landing spot follows the run kind unless `output_path` names one: a run over named ids writes
    the probe files, a limited run writes the smoke files, and a full run writes the label sidecar
    itself. A resumed run keeps the labels already in the output and labels only the titles missing
    from it.
    """
    input_path = Path(input_path)
    titles, conflicts = load_titles(input_path)

    todo = sorted(titles.items())
    if limit is not None:
        todo = todo[:limit]
    if ids_file is not None:
        wanted = json.loads(Path(ids_file).read_text())
        if isinstance(wanted, dict):
            wanted = wanted.get("recipe_ids", [])
        keep = {int(item) for item in wanted}
        todo = [item for item in todo if item[0] in keep]
        if not todo:
            raise ValueError(f"{ids_file} names no recipe in the pair set")

    smoke = limit is not None
    targeted = ids_file is not None
    out_path = PROBE_JSONL if targeted else (SMOKE_JSONL if smoke else OUTPUT_JSONL)
    manifest_path = PROBE_MANIFEST if targeted else (SMOKE_MANIFEST if smoke else OUTPUT_MANIFEST)
    if output_path is not None:
        # A dated output keeps a new full run from landing on top of the set it
        # replaces, so the two can be compared. The manifest follows the labels.
        out_path = Path(output_path)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[int, dict] = {}
    if resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                row = json.loads(line)
                done[row["recipe_id"]] = row
        logger.info("resuming with %d labels already written", len(done))

    pending = [(recipe_id, title) for recipe_id, title in todo if recipe_id not in done]
    client = LabellerClient(model_id=model_id)
    logger.info("labelling %d titles with %s (thinking=%s, temperature=%s)",
                len(pending), client.model_id, client.think, client.temperature)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    totals = Counter()
    failures: list[dict] = []

    def work(item: tuple[int, str]):
        recipe_id, title = item
        row, usage, error = label_one(recipe_id, title, client, created_at)
        return recipe_id, title, row, usage, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, item) for item in pending]
        for future in as_completed(futures):
            recipe_id, title, row, usage, error = future.result()
            totals["calls"] += usage["calls"]
            totals["retries"] += usage["retries"]
            totals["prompt_tokens"] += usage["prompt_tokens"]
            totals["output_tokens"] += usage["output_tokens"]
            if row is None:
                failures.append({"recipe_id": recipe_id, "name": title, "error": error})
                logger.warning("no label for %d (%s)", recipe_id, title)
                continue
            done[recipe_id] = row
            if smoke or targeted:
                print(f"{recipe_id:>5}  {row['category']:<10} {row['confidence']:<6} "
                      f"{row['name']}  ::  {row['evidence']}")

    rows = [done[recipe_id] for recipe_id in sorted(done)]
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # The full run is the label set the passes downstream read, so it is the one that gets
    # a pointer. The smoke and probe files are scratch, and are left untracked.
    if not smoke and not targeted:
        add_to_dvc(out_path)

    summary = summarise(rows)
    manifest = {
        "created_at": created_at,
        "run": "probe" if targeted else ("smoke" if smoke else "full"),
        "input": str(input_path),
        "output": str(out_path),
        "method": METHOD,
        "rule": "the boundary rule as written in the system prompt of this script",
        "labeller": {
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
        "titles": {
            "recipes_in_input": len(titles),
            "labelled_this_run": len(rows),
            "name_conflicts": conflicts,
        },
        "counts": summary,
        "usage": {
            "prompt_tokens": totals["prompt_tokens"],
            "output_tokens": totals["output_tokens"],
            "calls": totals["calls"],
            "retries": totals["retries"],
        },
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("wrote %d labels to %s", len(rows), out_path)
    print(f"labels: {summary['labels']}  " + "  ".join(
        f"{name}={count}" for name, count in sorted(summary["by_category"].items())))
    print("confidence: " + "  ".join(
        f"{name}={count}" for name, count in sorted(summary["by_confidence"].items())))
    print(f"flagged for review: {summary['flagged']}")
    if failures:
        print(f"failures: {len(failures)} (listed in {manifest_path})")
    return 0


# The file the labelling pass writes is the file the reviewed calls are applied to, so the apply
# pass reads its destination from the names above rather than spelling the same path twice.
SIDECAR_JSONL = OUTPUT_JSONL
SIDECAR_MANIFEST = OUTPUT_MANIFEST

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


def apply_category_calls(
    sidecar_path: Path = SIDECAR_JSONL,
    manifest_path: Path = SIDECAR_MANIFEST,
    dry_run: bool = False,
) -> int:
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
    sidecar_path = Path(sidecar_path)
    manifest_path = Path(manifest_path)

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

    if dry_run:
        print("dry run: nothing written")
        return 0

    temporary = sidecar_path.with_name(sidecar_path.name + ".tmp")
    temporary.write_text("".join(raw))
    temporary.replace(sidecar_path)

    # The settled calls have changed the sidecar, so its pointer has to move with them.
    add_to_dvc(sidecar_path)

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
