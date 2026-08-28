"""Drafts the user-question prompts half of the eval-data golden subset.

For each of nine query types, the drafter selects context chunks from the recipe
corpus using the build encoder for similarity, then asks a model to produce a
single user-style query string per selection. Output rows are written to
``data/eval/prompts.jsonl`` and carry the full context they were generated from,
so the answer-drafting step can re-use the same context without re-fetching.

The drafter is resumable: rows already present for a (query_type, primary_id)
pair are skipped on re-run. Each run starts by archiving any existing output to
``prompts.jsonl.bak.<timestamp>`` so an interrupted run leaves a recoverable copy.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Iterable

import json
import random
import shutil
import time

from ollama import chat
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from ..retrieval.chunker import (
    Chunk,
    _chunk_info,
    _chunk_it_ing,
    _chunk_whole,
)
from ..retrieval.dense.encoder import Encoder
from .preprocessor import Recipe


QUERY_TYPES: tuple[str, ...] = (
    "recipe",
    "technique",
    "recipe_with_a_swap",
    "info",
    "technique_suggestion",
    "recipe_suggestion_off_ingredients",
    "recipe_suggestion_off_instructions",
    "recipe_suggestion_off_info",
    "ingredient_suggestion",
)

TYPES_WITH_ONE_PRIMARY: frozenset[str] = frozenset(
    {"recipe", "technique", "info", "technique_suggestion"}
)

# A negative chunk_id reserved for selections that don't correspond to a chunker
# chunk (the per-recipe info section, which the chunker emits per-entry rather
# than per-section). Reserving a sentinel range keeps primary_id lookups stable.
INFO_SECTION_CHUNK_ID_BASE = -1


class _PromptRecord(BaseModel):
    """The single-string JSON shape the model is asked to emit."""

    query: str


@dataclass
class _Selection:
    """One (type, primary) selection, with the full context the model will see."""

    query_type: str
    primary_id: int
    primary_text: str
    primary_recipe_id: int
    sibling_texts: list[str]
    sibling_recipe_ids: list[int]


def _load_recipes(recipes_path: Path) -> list[Recipe]:
    """Read the cleaned-recipe corpus from disk."""
    with recipes_path.open() as f:
        return [Recipe(**r) for r in json.load(f)]


def _chunker_id_stream() -> count:
    """A fresh chunk_id iterator. Real chunks use non-negative ids; the
    info-section wrapper uses negative ids in ``INFO_SECTION_CHUNK_ID_BASE``
    so the two spaces never collide when sharing a corpus-level key."""
    return count()


def _build_whole_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return _chunk_whole(recipes, _chunker_id_stream())


def _build_ingredient_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return [c for c in _chunk_it_ing(recipes, _chunker_id_stream()) if c.chunk_type == "ingredients"]


def _build_instruction_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return [c for c in _chunk_it_ing(recipes, _chunker_id_stream()) if c.chunk_type == "instructions"]


def _build_info_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return _chunk_info(recipes, _chunker_id_stream())


@dataclass
class _InfoSection:
    """Per-recipe wrapper around the chunker's per-entry info chunks.

    The chunker emits one chunk per info entry; the ``info`` query type
    selects one chunk per recipe's complete info section. ``_InfoSection``
    joins the recipe's entries into a single block with a stable negative
    ``primary_id`` so resume-key lookups stay unique.
    """

    recipe_id: int
    text: str

    @property
    def primary_id(self) -> int:
        return INFO_SECTION_CHUNK_ID_BASE - self.recipe_id

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "_InfoSection | None":
        if not recipe.info:
            return None
        rendered = "\n\n".join(
            _render_info_entry(entry) for entry in recipe.info
        ).strip()
        if not rendered:
            return None
        return cls(recipe_id=recipe.recipe_id, text=rendered)


def _render_info_entry(entry: dict[str, Any]) -> str:
    """Render one info entry (group_name + content) the same way the chunker does."""
    name = entry.get("group_name")
    content = entry.get("content", "") or ""
    if name:
        return f"## {name}\n{content}"
    return content


def _build_info_sections(recipes: list[Recipe]) -> list[_InfoSection]:
    return [s for s in (_InfoSection.from_recipe(r) for r in recipes) if s is not None]


def _nearest_neighbour_ids(
    target_text: str,
    target_recipe_id: int,
    pool: list[Chunk],
    encoder: Encoder,
    k: int,
) -> list[int]:
    """Return up to ``k`` recipe_ids from ``pool`` whose chunks are most similar
    to ``target_text``. The pool's own recipe is excluded."""
    if not pool:
        return []
    pool = [c for c in pool if c.recipe_id != target_recipe_id]
    if not pool:
        return []
    target_vec = encoder.encode_query(target_text)
    pool_texts = [c.text for c in pool]
    pool_vecs = encoder._encode_documents(pool_texts)
    scores = pool_vecs @ target_vec
    order = scores.argsort()[::-1]
    ranked_recipes: list[int] = []
    seen: set[int] = set()
    for idx in order:
        rid = pool[idx].recipe_id
        if rid in seen:
            continue
        seen.add(rid)
        ranked_recipes.append(rid)
        if len(ranked_recipes) >= k:
            break
    return ranked_recipes


def _recipe_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    """Render a whole slice using the same shaper the chunker uses for whole chunks."""
    for c in _build_whole_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _ingredient_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _build_ingredient_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _instruction_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _build_instruction_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _info_section_text(sections_by_primary: dict[int, _InfoSection], primary_id: int) -> str:
    section = sections_by_primary.get(primary_id)
    if section is None:
        raise KeyError(f"info section {primary_id} not found")
    return section.text


def _selections_for_type(
    query_type: str,
    per_type_count: int,
    recipes: list[Recipe],
    encoder: Encoder,
    rng: random.Random,
) -> list[_Selection]:
    """Return the selections (one per row) for a single query type."""
    recipes_by_id = {r.recipe_id: r for r in recipes}

    if query_type == "recipe":
        chunk_pool = _build_whole_chunks(recipes)
        picks = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        return [
            _Selection(query_type=query_type, primary_id=c.chunk_id,
                       primary_text=c.text, primary_recipe_id=c.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for c in picks
        ]

    if query_type == "technique":
        # Same selection as recipe; the model is told to find a notable technique.
        chunk_pool = _build_whole_chunks(recipes)
        picks = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        return [
            _Selection(query_type=query_type, primary_id=c.chunk_id,
                       primary_text=c.text, primary_recipe_id=c.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for c in picks
        ]

    if query_type == "technique_suggestion":
        chunk_pool = _build_whole_chunks(recipes)
        picks = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        return [
            _Selection(query_type=query_type, primary_id=c.chunk_id,
                       primary_text=c.text, primary_recipe_id=c.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for c in picks
        ]

    if query_type == "info":
        sections = _build_info_sections(recipes)
        picks = rng.sample(sections, k=min(per_type_count, len(sections)))
        return [
            _Selection(query_type=query_type, primary_id=s.primary_id,
                       primary_text=s.text, primary_recipe_id=s.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for s in picks
        ]

    if query_type == "recipe_with_a_swap":
        chunk_pool = _build_whole_chunks(recipes)
        seeds = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        selections: list[_Selection] = []
        for seed in seeds:
            siblings = _nearest_neighbour_ids(
                target_text=seed.text,
                target_recipe_id=seed.recipe_id,
                pool=chunk_pool,
                encoder=encoder,
                k=2,
            )
            if len(siblings) < 2:
                continue
            # Pre-select the closest sibling as the swap target; the other sibling
            # is the reference. The model only writes the prompt.
            swap_recipe_id = siblings[0]
            other_recipe_id = siblings[1]
            selections.append(_Selection(
                query_type=query_type,
                primary_id=seed.chunk_id,
                primary_text=seed.text,
                primary_recipe_id=seed.recipe_id,
                sibling_texts=[
                    _recipe_text(recipes_by_id, other_recipe_id),
                    _recipe_text(recipes_by_id, swap_recipe_id),
                ],
                sibling_recipe_ids=[other_recipe_id, swap_recipe_id],
            ))
        return selections

    if query_type == "ingredient_suggestion":
        chunk_pool = _build_whole_chunks(recipes)
        seeds = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        selections = []
        for seed in seeds:
            siblings = _nearest_neighbour_ids(
                target_text=seed.text,
                target_recipe_id=seed.recipe_id,
                pool=chunk_pool,
                encoder=encoder,
                k=2,
            )
            if len(siblings) < 2:
                continue
            selections.append(_Selection(
                query_type=query_type,
                primary_id=seed.chunk_id,
                primary_text=seed.text,
                primary_recipe_id=seed.recipe_id,
                sibling_texts=[
                    _recipe_text(recipes_by_id, rid) for rid in siblings
                ],
                sibling_recipe_ids=list(siblings),
            ))
        return selections

    if query_type == "recipe_suggestion_off_ingredients":
        chunk_pool = _build_ingredient_chunks(recipes)
        pool_to_recipe = {c.chunk_id: c.recipe_id for c in chunk_pool}
        return _suggestion_selections(
            query_type=query_type,
            per_type_count=per_type_count,
            encoder=encoder,
            rng=rng,
            chunk_id_to_recipe_id=pool_to_recipe,
            chunk_pool=chunk_pool,
            chunk_lookup=lambda rid: _ingredient_text(recipes_by_id, rid),
        )

    if query_type == "recipe_suggestion_off_instructions":
        chunk_pool = _build_instruction_chunks(recipes)
        pool_to_recipe = {c.chunk_id: c.recipe_id for c in chunk_pool}
        return _suggestion_selections(
            query_type=query_type,
            per_type_count=per_type_count,
            encoder=encoder,
            rng=rng,
            chunk_id_to_recipe_id=pool_to_recipe,
            chunk_pool=chunk_pool,
            chunk_lookup=lambda rid: _instruction_text(recipes_by_id, rid),
        )

    if query_type == "recipe_suggestion_off_info":
        sections = _build_info_sections(recipes)
        if len(sections) < per_type_count * 3:
            return []
        sections_by_primary = {s.primary_id: s for s in sections}
        sections_by_recipe = {s.recipe_id: s for s in sections}
        seeds = rng.sample(sections, k=min(per_type_count, len(sections)))
        selections = []
        for seed in seeds:
            siblings = _nearest_neighbour_ids(
                target_text=seed.text,
                target_recipe_id=seed.recipe_id,
                pool=[_to_chunk(section) for section in sections],
                encoder=encoder,
                k=2,
            )
            if len(siblings) < 2:
                continue
            selections.append(_Selection(
                query_type=query_type,
                primary_id=seed.primary_id,
                primary_text=seed.text,
                primary_recipe_id=seed.recipe_id,
                sibling_texts=[
                    _info_section_text(sections_by_primary, sections_by_recipe[s].primary_id)
                    for s in siblings
                ],
                sibling_recipe_ids=list(siblings),
            ))
        return selections

    raise ValueError(f"unknown query_type: {query_type}")


def _to_chunk(section: _InfoSection) -> Chunk:
    """Coerce an ``_InfoSection`` into a ``Chunk`` so it can flow through
    nearest-neighbour selection. Uses the same negative primary_id."""
    return Chunk(
        chunk_id=section.primary_id,
        recipe_id=section.recipe_id,
        chunk_type="info_section",
        text=section.text,
    )


def _suggestion_selections(
    query_type: str,
    per_type_count: int,
    encoder: Encoder,
    rng: random.Random,
    chunk_id_to_recipe_id: dict[int, int],
    chunk_pool: list[Chunk],
    chunk_lookup,
) -> list[_Selection]:
    """Shared selection logic for the three ``recipe_suggestion_off_*`` types.

    Each selection is 1 seed chunk + 2 nearest-neighbour chunks from distinct
    recipes, all from the same chunk-type pool. The pool's own
    ``chunk_id_to_recipe_id`` map is passed in because ids are stream-local.
    """
    if len(chunk_pool) < per_type_count * 3:
        return []
    seeds = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
    selections: list[_Selection] = []
    for seed in seeds:
        siblings = _nearest_neighbour_ids(
            target_text=seed.text,
            target_recipe_id=seed.recipe_id,
            pool=chunk_pool,
            encoder=encoder,
            k=2,
        )
        if len(siblings) < 2:
            continue
        selections.append(_Selection(
            query_type=query_type,
            primary_id=seed.chunk_id,
            primary_text=seed.text,
            primary_recipe_id=chunk_id_to_recipe_id[seed.chunk_id],
            sibling_texts=[chunk_lookup(rid) for rid in siblings],
            sibling_recipe_ids=list(siblings),
        ))
    return selections


# --- Per-type prompt templates ---------------------------------------------


_SYSTEM_PROMPT = (
    "You are a home cook drafting questions to test a recipe retrieval system. "
    "Produce exactly one user-style query string. Your entire response must be "
    "a single JSON object on one line - no markdown fences, no commentary, "
    'no surrounding text. Exact shape: {"query": "<the query>"}'
)


def _user_message(query_type: str, primary_text: str, sibling_texts: list[str]) -> str:
    """Build the user-role message body for a single (type, primary) selection."""
    if query_type in TYPES_WITH_ONE_PRIMARY:
        body = _single_block_user_message(query_type, primary_text)
    else:
        body = _multi_block_user_message(query_type, primary_text, sibling_texts)
    return (
        body
        + "\n\nRespond with a single JSON object on one line - no markdown fences, "
        + 'no commentary: {"query": "<the query>"}'
    )


def _single_block_user_message(query_type: str, primary_text: str) -> str:
    label_map = {
        "recipe": "RECIPE",
        "technique": "RECIPE",
        "technique_suggestion": "RECIPE",
        "info": "INFO SECTION",
    }
    label = label_map[query_type]
    instruction_map = {
        "recipe": (
            "Generate a short question a home cook might type to find this exact recipe. "
            "Reference what makes it distinctive (an ingredient, technique, cuisine, or occasion)."
        ),
        "technique": (
            "Read the instructions. Pick one notable technique they describe and ask how to do it. "
            "Don't quote the recipe name verbatim."
        ),
        "technique_suggestion": (
            "Frame a how-to question as if you're seeking advice for a real kitchen situation. "
            "Do not name this recipe or assume the reader has seen it."
        ),
        "info": (
            "Generate a meta question about the section above — spiciness, cuisine, difficulty, "
            "timing, or serving. Don't reference the recipe by name."
        ),
    }
    return f"{label}:\n{primary_text}\n\n{instruction_map[query_type]}"


def _multi_block_user_message(query_type: str, primary_text: str, sibling_texts: list[str]) -> str:
    if query_type == "recipe_with_a_swap":
        return (
            "Below are three similar recipes. The first is the primary; the second is a reference; "
            "the third is the swap target. Write a prompt that asks the home cook how to substitute "
            "one ingredient from the primary with something from the swap target, using the reference "
            "to keep the swap grounded.\n\n"
            f"PRIMARY:\n{primary_text}\n\n"
            f"REFERENCE:\n{sibling_texts[0]}\n\n"
            f"SWAP TARGET:\n{sibling_texts[1]}"
        )
    if query_type == "ingredient_suggestion":
        return (
            "Below are three recipes. Write a prompt that asks for ingredient substitutions that would "
            "work across all three. Anchor the question in something the recipes share.\n\n"
            f"RECIPE 1:\n{primary_text}\n\n"
            f"RECIPE 2:\n{sibling_texts[0]}\n\n"
            f"RECIPE 3:\n{sibling_texts[1]}"
        )
    # The three recipe_suggestion_off_* types.
    label_map = {
        "recipe_suggestion_off_ingredients": "shared ingredients",
        "recipe_suggestion_off_instructions": "shared cooking method",
        "recipe_suggestion_off_info": "shared notes or meta info",
    }
    return (
        f"Below are three chunks from three different recipes that share {label_map[query_type]}. "
        "Write a 'what can I make' prompt grounded in the overlap — don't name a specific recipe.\n\n"
        f"CHUNK 1:\n{primary_text}\n\n"
        f"CHUNK 2:\n{sibling_texts[0]}\n\n"
        f"CHUNK 3:\n{sibling_texts[1]}"
    )


def _prompt_template_tag(query_type: str) -> str:
    return f"{query_type}_v1"


# --- Ollama call + resume + write ------------------------------------------


def _call_model(
    query_type: str,
    primary_text: str,
    sibling_texts: list[str],
    model_id: str,
    max_retries: int,
    backoff_base: float,
) -> str:
    """Call the model and return the parsed query string, or "" on failure."""
    user_msg = _user_message(query_type, primary_text, sibling_texts)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
            )
            content = _strip_markdown_fence(response.message.content)
            parsed = _PromptRecord.model_validate_json(content)
            return parsed.query.strip()
        except (ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                # Jitter so parallel workers don't retry in lockstep.
                sleep_for = (backoff_base ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
    if last_error is not None:
        print(f"_call_model failed for type={query_type}: {last_error}")
    return ""


def _strip_markdown_fence(content: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence if present.

    Some cloud models wrap their JSON in a markdown code block even when the
    request says ``format="json"``. Pydantic's strict parser rejects the fence,
    so we strip it before validating.
    """
    s = content.strip()
    if s.startswith("```"):
        # Drop the opening fence line (```json or ```), then any closing fence.
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


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


def _archive_output(output_path: Path) -> None:
    """Move an existing output file to a timestamped backup before truncation."""
    if not output_path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = output_path.with_name(f"{output_path.name}.bak.{timestamp}")
    shutil.move(str(output_path), str(backup))


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


def _selection_to_row(
    selection: _Selection,
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
        "prompt_template": _prompt_template_tag(selection.query_type),
        "source": "llm-draft-prompt",
        "prompt_verified": False,
        # Internal field used by resume; downstream tooling can ignore.
        "_primary_id": selection.primary_id,
    }


def _context_chunks_for_selection(selection: _Selection) -> dict[str, Any]:
    """Build the ``context_chunks`` block stored on each row."""
    if selection.query_type == "info":
        return {"info_chunk": selection.primary_text}
    if selection.query_type == "recipe_with_a_swap":
        # Other sibling is reference; the second sibling is the swap target.
        return {
            "primary": selection.primary_text,
            "swap_targets": selection.sibling_texts[1:2],
        }
    if selection.query_type in TYPES_WITH_ONE_PRIMARY:
        return {"primary": selection.primary_text}
    # recipe_suggestion_off_* and ingredient_suggestion.
    return {
        "primary": selection.primary_text,
        "siblings": selection.sibling_texts,
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
    is skipped. The function moves an existing output to a timestamped backup
    before writing the new file, and writes a sidecar of failed selections so
    resume can skip past known-bad keys instead of re-trying them forever.
    """
    recipes = _load_recipes(recipes_path)
    if not recipes:
        raise ValueError(f"no recipes in {recipes_path}")

    rng = random.Random(seed)

    selections: list[_Selection] = []
    for query_type in QUERY_TYPES:
        selections.extend(_selections_for_type(
            query_type=query_type,
            per_type_count=per_type_count,
            recipes=recipes,
            encoder=encoder,
            rng=rng,
        ))

    processed = _load_processed_keys(output_path)
    failed = _load_failed_keys(output_path)
    _archive_output(output_path)
    _archive_failed(output_path)

    rows: list[dict[str, Any]] = []
    new_failed: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sel = {
            executor.submit(
                _call_model,
                sel.query_type,
                sel.primary_text,
                sel.sibling_texts,
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

    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

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


def _archive_failed(output_path: Path) -> None:
    """Move an existing failed sidecar to a timestamped backup before truncation."""
    failed_path = _failed_path(output_path)
    if not failed_path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = failed_path.with_name(f"{failed_path.name}.bak.{timestamp}")
    shutil.move(str(failed_path), str(backup))
