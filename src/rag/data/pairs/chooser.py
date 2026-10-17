import hashlib
import json
import logging
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from ...evaluation.harness import load_snapshot, resolve_snapshot_folder
from ..recipe_format import recipe_to_string
from .manifest import add_to_dvc
from .rendering import strip_markdown_fence
from .selection import resolve_snapshot_corpus

logger = logging.getLogger(__name__)

# The pool this run reads, and where the pairs land. A limited run writes the
# smoke pair instead, so a prompt change never lands on top of the real set.
POOL_JSONL = Path("data/pairs/candidate_pool.jsonl")
PAIRS_JSONL = Path("data/pairs/recipe_pairs.jsonl")
PAIRS_MANIFEST = Path("data/pairs/recipe_pairs_manifest.json")
SMOKE_JSONL = Path("data/pairs/chooser_smoke.jsonl")
SMOKE_MANIFEST = Path("data/pairs/chooser_smoke_manifest.json")

# A run over a named subset of primaries writes here. It exists so a re-check of
# specific calls can be made without disturbing the smoke pair or the full set.
PROBE_JSONL = Path("data/pairs/chooser_probe.jsonl")
PROBE_MANIFEST = Path("data/pairs/chooser_probe_manifest.json")

# The chooser's own settings: greedy, seeded, and thinking off. The measured call
# with thinking on spent about 4,500 output tokens on its own reasoning, roughly
# ten times the cost of the answer, and the judgement this call makes is a
# comparison of two ingredient lists rather than a derivation.
MODEL_ID = "qwen3.5:397b-cloud"
TEMPERATURE = 0.0
SEED = 42
THINK = False
NUM_CTX = 16384
NUM_PREDICT = 6000

# Sampling retries. A call that fails to parse, validate, or name an offered
# rank is retried with backoff; a primary whose retries all fail is reported in
# the manifest rather than quietly dropped from the set.
MAX_RETRIES = 3
BACKOFF_BASE = 2

# The provider's per-minute cap, and the rate the run is priced at.
MAX_WORKERS = 3
PRICE_PER_M_INPUT = 0.60
PRICE_PER_M_OUTPUT = 3.60

# How many partners each call picks, and how many reasons a smoke run prints.
PICKS_PER_CALL = 2

SYSTEM_PROMPT = """You choose two recipes that should each be fused with a primary recipe to make a new dish.

You will be given the primary recipe, then a numbered band of candidate recipes, then a short list of numbered anchors. The band is what the new dishes must be built from. The anchors exist only for the case where nothing in the band works.

Choose two candidates from the band, each making a different pair with the primary:
- exactly two picks, never one and never three, and both ranks must come from the band
- the two picks must be different recipes
- prefer two picks that differ from each other in kind, so the two pairs are not two versions of the same dish
- reject a candidate that is the same dish as the primary or a near copy of it
- reject a component rather than a dish: a sauce, a dressing, a marinade, a salad, a side, a drink
- reject a pairing whose result no cook would attempt
- if every candidate in the band must be rejected, set "fallback" to true and return two ranks from the anchors instead, or the single anchor when only one is listed, and say what was wrong with the band

Give one sentence of reason per pick, naming what the two recipes share and what that new dish becomes. No prose beyond those sentences.

Reply with JSON only:
{"picks": [{"rank": <first rank>, "reason": "<one sentence naming what the two share and what the new dish becomes>"}, {"rank": <second rank>, "reason": "<the same, for your second pick>"}], "fallback": false}

Everything inside angle brackets is a placeholder to be replaced. Return two actual ranks from the band above, each with a real sentence."""


class Pick(BaseModel):
    """One partner the chooser named, with the sentence that justifies it."""

    rank: int
    reason: str = Field(min_length=1, max_length=600)


class Choice(BaseModel):
    """The chooser's JSON response."""

    picks: list[Pick]
    fallback: bool = False


def load_inputs(snapshot_target: str = "latest_build", limit: int | None = None,
                primaries_file: str | None = None) -> dict:
    """The pool, the corpus it points into, and the prompt the run will send."""
    if not POOL_JSONL.exists():
        raise FileNotFoundError(f"{POOL_JSONL} is missing; build the pool first")
    pool = [json.loads(line) for line in POOL_JSONL.open() if line.strip()]

    snapshot_folder = resolve_snapshot_folder(snapshot_target)
    snapshot = load_snapshot(snapshot_folder)
    resolved, entry = resolve_snapshot_corpus(snapshot)
    recipes = json.load(open(resolved.path))

    primaries = pool
    if limit is not None:
        primaries = sorted(pool, key=lambda r: r["primary_id"])[:limit]
    if primaries_file is not None:
        wanted = json.loads(Path(primaries_file).read_text())
        if isinstance(wanted, dict):
            wanted = wanted.get("primary_ids", [])
        keep = {int(item) for item in wanted}
        primaries = [p for p in pool if p["primary_id"] in keep]
        if not primaries:
            raise ValueError(f"{primaries_file} names no primary in the pool")
    return {
        "pool": pool,
        "primaries": primaries,
        "recipes": {r["recipe_id"]: r for r in recipes},
        "snapshot_folder": snapshot_folder,
        "dataset_key": snapshot.dataset_key,
        "dataset_path": str(resolved.path),
        "dataset_checksum": resolved.checksum,
        "dataset_registered_at": entry.get("created_at"),
        "pool_rows": len(pool),
    }


def ingredient_line(recipe: dict) -> str:
    """A candidate's ingredients as one comma-separated line, names only."""
    names = [
        item["name"].strip()
        for group in recipe.get("ingredient_groups", [])
        for item in group.get("ingredients", [])
        if item.get("name")
    ]
    return ", ".join(names) if names else "no ingredients listed"


def offered(primary_row: dict) -> tuple[list[dict], list[dict]]:
    """The band and the anchors the chooser may see, with flagged rows dropped.

    Excluding a flagged candidate here rather than describing it in the prompt is
    what keeps a near-copy from being returned by a model that reads the rule
    loosely.
    """
    band = [c for c in primary_row["candidates"] if c["role"] == "band" and not c["repeat"]]
    anchors = [c for c in primary_row["candidates"] if c["role"] == "anchor" and not c["repeat"]]
    return band, anchors


def user_message(primary_name: str, primary_text: str, band: list[dict], anchors: list[dict],
                 recipes: dict[int, dict]) -> str:
    """The call's user turn: the primary in full, then the two numbered lists."""
    lines = ["PRIMARY RECIPE", primary_text, ""]
    if band:
        lines.append(f"BAND (ranks {band[0]['rank']} to {band[-1]['rank']}, exactly one pick must come from here)")
        for candidate in band:
            lines.append(f"{candidate['rank']}. {candidate['name']}  (cos {candidate['cosine']:.3f})")
            lines.append(f"    Ingredients: {ingredient_line(recipes.get(candidate['recipe_id'], {}))}")
    else:
        lines.append("BAND (empty: every candidate was rejected as a repeat, so this must be a fallback)")
    lines.append("")
    if anchors:
        lines.append("ANCHORS (fallback only)")
        for candidate in anchors:
            lines.append(f"{candidate['rank']}. {candidate['name']}  (cos {candidate['cosine']:.3f})")
    else:
        lines.append("ANCHORS (none available)")
    return "\n".join(lines)


def prompt_hash() -> str:
    """A stable fingerprint of the prompt the run sends, for the manifest."""
    return "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


class ChooserClient:
    """The chat call, its retries, and the tokens it spends.

    The retry covers the failures the provider produces in practice: a response
    that arrives wrapped in a markdown fence, a response that does not validate
    against the response shape, and picks that repeat a rank, name a rank the
    chooser was not offered, or arrive in the wrong number.
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
        """One call, returned whole, for the probe that inspects the wire shape."""
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

    def choose(self, message: str, band_ranks: set[int], anchor_ranks: set[int]):
        """The validated picks, with the tokens and the latency they took.

        A call that names more picks than needed, or names one that cannot be
        used, keeps the usable ones and records that it did, so a slightly loose
        response is not thrown away and not silently passed off as clean either.
        Fewer usable picks than asked for is a failure and gets retried.
        """
        usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0,
                 "picks_returned": 0, "trimmed": 0}
        started = time.monotonic()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.raw_call(message)
                usage["calls"] += 1
                usage["prompt_tokens"] += response.prompt_eval_count or 0
                usage["output_tokens"] += response.eval_count or 0
                choice = Choice.model_validate_json(strip_markdown_fence(response.message.content or ""))

                allowed = anchor_ranks if choice.fallback else band_ranks
                wanted = min(PICKS_PER_CALL, len(allowed))
                usable: list[Pick] = []
                for pick in choice.picks:
                    if pick.rank in allowed and pick.reason.strip() and \
                            not any(kept.rank == pick.rank for kept in usable):
                        usable.append(pick)
                if len(usable) < wanted:
                    kind = "anchor" if choice.fallback else "band"
                    raise ValueError(f"{len(usable)} usable {kind} picks, {wanted} wanted")

                usage["picks_returned"] = len(choice.picks)
                usage["trimmed"] += int(len(usable) != len(choice.picks))
                usage["seconds"] = round(time.monotonic() - started, 2)
                return usable[:wanted], bool(choice.fallback), usage
            except (ValidationError, ValueError, Exception) as error:  # noqa: BLE001
                last_error = error
                usage["retries"] += 1
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("chooser call failed: %s", last_error)
        usage["error"] = str(last_error)
        usage["seconds"] = round(time.monotonic() - started, 2)
        return None, None, usage


def choose_for(primary_row: dict, inputs: dict, client: ChooserClient) -> tuple[list[dict], dict, str | None]:
    """One primary's pair rows, or the error that stopped it."""
    recipes = inputs["recipes"]
    primary = recipes.get(primary_row["primary_id"])
    if primary is None:
        empty = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0,
                 "picks_returned": 0, "trimmed": 0, "seconds": 0.0}
        return [], empty, f"recipe {primary_row['primary_id']} is not in the corpus"

    band, anchors = offered(primary_row)
    message = user_message(primary_row["primary_name"], recipe_to_string(primary), band, anchors, recipes)
    picks, fallback, usage = client.choose(message, {c["rank"] for c in band}, {c["rank"] for c in anchors})
    if picks is None:
        return [], usage, usage.get("error", "every attempt failed")

    by_rank = {c["rank"]: c for c in (anchors if fallback else band)}
    rows = []
    for index, pick in enumerate(picks, start=1):
        partner = by_rank[pick.rank]
        others = [p.rank for p in picks if p.rank != pick.rank]
        rows.append({
            "primary_id": primary_row["primary_id"],
            "primary_name": primary_row["primary_name"],
            "partner_id": partner["recipe_id"],
            "partner_name": partner["name"],
            "partner_rank": partner["rank"],
            "partner_cosine": partner["cosine"],
            "pair_index": index,
            "pair_index_of": len(picks),
            "other_partner_rank": others[0] if others else None,
            "other_partner_id": by_rank[others[0]]["recipe_id"] if others else None,
            "band": primary_row["band"],
            "anchors": [
                {"rank": c["rank"], "recipe_id": c["recipe_id"], "name": c["name"], "cosine": c["cosine"]}
                for c in anchors
            ],
            "band_size_offered": len(band),
            "fallback": bool(fallback),
            "reason": pick.reason.strip(),
            "space": primary_row["space"],
            "encoder": primary_row["encoder"],
            "snapshot": inputs["snapshot_folder"],
            "dataset_key": inputs["dataset_key"],
            "chooser_model": client.model_id,
            "temperature": client.temperature,
            "seed": client.seed,
            "thinking": client.think,
            "picks_returned": usage["picks_returned"],
            "picks_trimmed": bool(usage["trimmed"]),
        })
    return rows, usage, None


def choose_pairs(
    snapshot_target: str = "latest_build",
    limit: int | None = None,
    primaries_file: str | None = None,
    output_path: Path | None = None,
    workers: int = MAX_WORKERS,
    resume: bool = False,
) -> int:
    """Run the pass: read the pool, choose for every primary, write the pairs.

    The landing spot follows the run kind unless `output_path` names one: a run over named primaries
    writes the probe pair, a limited run writes the smoke pair, and a full run writes the pair set
    itself. A resumed run keeps the rows already in the output and chooses only for the primaries
    missing from it.
    """
    inputs = load_inputs(snapshot_target, limit, primaries_file)
    client = ChooserClient()
    smoke = limit is not None
    targeted = primaries_file is not None
    out_path = PROBE_JSONL if targeted else (SMOKE_JSONL if smoke else PAIRS_JSONL)
    manifest_path = PROBE_MANIFEST if targeted else (SMOKE_MANIFEST if smoke else PAIRS_MANIFEST)
    if output_path is not None:
        # A dated output keeps a new full run from landing on top of the set it
        # replaces, so the two can be compared. The manifest follows the pairs.
        out_path = output_path
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[int, list[dict]] = {}
    if resume and out_path.exists():
        for line in out_path.open():
            if line.strip():
                row = json.loads(line)
                done.setdefault(row["primary_id"], []).append(row)
        logger.info("resuming with %d pairs across %d primaries already written",
                    sum(len(v) for v in done.values()), len(done))

    todo = [p for p in inputs["primaries"] if p["primary_id"] not in done]
    logger.info("choosing for %d primaries with %s (thinking=%s, temperature=%s)",
                len(todo), client.model_id, client.think, client.temperature)

    totals = Counter()
    failures: list[dict] = []
    lock = threading.Lock()
    started = time.monotonic()

    def work(primary_row: dict):
        rows, usage, error = choose_for(primary_row, inputs, client)
        return primary_row, rows, usage, error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for count, future in enumerate(as_completed(futures), start=1):
            primary_row, new_rows, usage, error = future.result()
            with lock:
                for key, value in usage.items():
                    totals[key] += value
                if not new_rows:
                    failures.append({"primary_id": primary_row["primary_id"],
                                     "primary_name": primary_row["primary_name"], "error": error})
                    logger.warning("no pair for %s: %s", primary_row["primary_id"], error)
                else:
                    done[primary_row["primary_id"]] = new_rows
                    totals["fallbacks"] += int(new_rows[0]["fallback"])
                    if count % 25 == 0 or count == len(todo):
                        logger.info("%d/%d chosen, %d pairs, %d fallbacks, %d failures",
                                    count, len(todo), sum(len(v) for v in done.values()),
                                    totals["fallbacks"], len(failures))

    rows = [row for key in sorted(done) for row in sorted(done[key], key=lambda r: r["pair_index"])]
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # The full run is the pair set the passes downstream read, so it is the one that gets
    # a pointer. The smoke and probe files are scratch, and are left untracked.
    if not smoke and not targeted:
        add_to_dvc(out_path)

    cost = (totals["prompt_tokens"] / 1_000_000) * PRICE_PER_M_INPUT + \
           (totals["output_tokens"] / 1_000_000) * PRICE_PER_M_OUTPUT
    distinct_partners = len({r["partner_id"] for r in rows})
    elapsed = round(time.monotonic() - started, 1)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": "probe" if targeted else ("smoke" if smoke else "full"),
        "pool": str(POOL_JSONL),
        "pool_rows": inputs["pool_rows"],
        "output": str(out_path),
        "primaries_file": str(primaries_file) if targeted else None,
        "chooser": {
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
        "dataset_key": inputs["dataset_key"],
        "dataset_path": inputs["dataset_path"],
        "dataset_checksum": inputs["dataset_checksum"],
        "dataset_registered_at": inputs["dataset_registered_at"],
        "snapshot": inputs["snapshot_folder"],
        "counts": {
            "attempted": len(todo) + len(done) if resume else len(inputs["primaries"]),
            "primaries": len({r["primary_id"] for r in rows}),
            "pairs": len(rows),
            "pairs_per_primary": dict(sorted(Counter(len(v) for v in done.values()).items())),
            "fallbacks": totals["fallbacks"],
            "failures": len(failures),
            "distinct_partners": distinct_partners,
            "picks_trimmed": int(totals["trimmed"]),
        },
        "partner_ranks": dict(sorted(Counter(r["partner_rank"] for r in rows).items())),
        "usage": {
            "prompt_tokens": totals["prompt_tokens"],
            "output_tokens": totals["output_tokens"],
            "calls": totals["calls"],
            "retries": totals["retries"],
            "wall_clock_seconds": elapsed,
            "cost_usd": round(cost, 4),
            "rates": {"input_per_m": PRICE_PER_M_INPUT, "output_per_m": PRICE_PER_M_OUTPUT},
        },
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("wrote %d pairs to %s", len(rows), out_path)
    logger.info("%d fallbacks, %d failures, %d prompt + %d output tokens, $%.4f, %.1fs",
                totals["fallbacks"], len(failures), totals["prompt_tokens"], totals["output_tokens"],
                cost, elapsed)
    if smoke:
        print_reasons(rows, failures)
    return 0


def print_reasons(rows: list[dict], failures: list[dict]) -> None:
    """Print every reason a smoke run produced, for reading rather than counting."""
    print(f"\n{len(rows)} pairs from {len({r['primary_id'] for r in rows})} primaries\n")
    print(f"{'primary':<32} {'#':<2} {'partner (rank, cos)':<42} {'fb':<3} reason")
    for row in rows:
        partner = f"{row['partner_name'][:30]} ({row['partner_rank']}, {row['partner_cosine']:.3f})"
        print(f"{row['primary_name'][:31]:<32} {row['pair_index']:<2} {partner:<42} "
              f"{'yes' if row['fallback'] else 'no':<3} {row['reason']}")
    if failures:
        print("\nfailed:")
        for failure in failures:
            print(f"  {failure['primary_id']} {failure['primary_name'][:40]}: {failure['error']}")


def probe_chooser(snapshot_target: str = "latest_build", limit: int | None = None,
                  primaries_file: str | None = None) -> int:
    """One call, printed whole, to check the wire shape before any batch."""
    inputs = load_inputs(snapshot_target, limit, primaries_file)
    client = ChooserClient()
    primary_row = sorted(inputs["primaries"], key=lambda r: r["primary_id"])[0]
    band, anchors = offered(primary_row)
    message = user_message(primary_row["primary_name"], recipe_to_string(inputs["recipes"][primary_row["primary_id"]]),
                           band, anchors, inputs["recipes"])
    print(f"model={client.model_id} think={client.think} temperature={client.temperature} seed={client.seed}")
    print(f"band offered={len(band)} anchors offered={len(anchors)}")
    started = time.monotonic()
    response = client.raw_call(message)
    elapsed = time.monotonic() - started
    thinking = response.message.thinking or ""
    content = response.message.content or ""
    print(f"elapsed={elapsed:.1f}s done_reason={response.done_reason}")
    print(f"prompt_eval_count={response.prompt_eval_count} eval_count={response.eval_count}")
    print(f"thinking chars={len(thinking)} content chars={len(content)}")
    print(f"thinking head: {thinking[:400]}")
    print(f"content:\n{strip_markdown_fence(content)}")
    try:
        choice = Choice.model_validate_json(strip_markdown_fence(content))
        for pick in choice.picks:
            print(f"parsed: rank={pick.rank} fallback={choice.fallback} reason={pick.reason}")
    except ValidationError as error:
        print(f"PARSE FAILED: {error}")
    return 0
