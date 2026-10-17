import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.chunker import (
    instruction_chunks,
    ingredient_chunks,
    whole_recipe_chunks,
)
from ...domain.types import Recipe
from ..prompts_drafter.selection import _InfoSection, _build_info_sections
from ..prompts_drafter.types import TYPES_WITH_ONE_PRIMARY


# Chunker entry points for the chunk types the drafter needs to look up.
_WHOLE_CHUNKER = whole_recipe_chunks
_INGREDIENT_CHUNKER = ingredient_chunks
_INSTRUCTION_CHUNKER = instruction_chunks


@dataclass
class AnswerSelection:
    """One verified-prompt row, enriched with the chunks the answer model sees.

    ``primary_text`` is taken from the row's own ``context_chunks.primary``
    (the drafter that produced the prompts already wrote the full text
    there). ``sibling_texts`` is built fresh from the cleaned-recipe corpus
    so the answer model has the other ground-truth recipes in front of it.
    For single-recipe types, ``sibling_texts`` stays empty.
    """

    query: str
    query_type: str
    prompt_template: str
    primary_id: int
    primary_recipe_id: int
    primary_text: str
    recipe_ids: list[int]
    source_prompt: str
    context_chunks: dict[str, Any] = field(default_factory=dict)
    sibling_texts: list[str] = field(default_factory=list)
    sibling_recipe_ids: list[int] = field(default_factory=list)
    # recipe_with_a_swap uses semantic labels instead of positional siblings.
    reference_text: str | None = None
    swap_text: str | None = None


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Read the verified prompt rows from disk."""
    with prompts_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _whole_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    """Render a whole-recipe chunk for one recipe."""
    for c in _WHOLE_CHUNKER([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _ingredient_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _INGREDIENT_CHUNKER([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _instruction_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _INSTRUCTION_CHUNKER([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _info_section_text(
    sections_by_recipe: dict[int, _InfoSection], recipe_id: int,
) -> str:
    section = sections_by_recipe.get(recipe_id)
    if section is None:
        raise KeyError(f"info section for recipe {recipe_id} not found")
    return section.text


def _primary_chunk_type(query_type: str) -> str:
    """Return the chunk type the primary of this row was emitted as.

    Mirrors the selection logic in the prompts drafter so the sibling
    chunks loaded for the other recipe_ids use the same chunker.
    """
    if query_type in {
        "recipe", "technique", "recipe_with_a_swap",
        "recipe_variation", "recipe_combination", "recipe_from_constraints",
        "recipe_adaptation_method",
    }:
        return "whole"
    if query_type == "info":
        return "info"
    if query_type == "recipe_suggestion_off_ingredients":
        return "ingredient"
    if query_type == "recipe_suggestion_off_instructions":
        return "instruction"
    if query_type == "recipe_suggestion_off_info":
        return "info"
    raise ValueError(f"unknown query_type: {query_type}")


def _text_for_recipe(
    query_type: str,
    recipe_id: int,
    recipes_by_id: dict[int, Recipe],
    info_sections_by_recipe: dict[int, _InfoSection],
) -> str:
    """Render the chunk text for one (query_type, recipe_id) pair."""
    chunk_type = _primary_chunk_type(query_type)
    if chunk_type == "whole":
        return _whole_text(recipes_by_id, recipe_id)
    if chunk_type == "ingredient":
        return _ingredient_text(recipes_by_id, recipe_id)
    if chunk_type == "instruction":
        return _instruction_text(recipes_by_id, recipe_id)
    if chunk_type == "info":
        return _info_section_text(info_sections_by_recipe, recipe_id)
    raise ValueError(f"unknown chunk type: {chunk_type}")


def selections_from_prompts(
    prompts: list[dict[str, Any]],
    recipes: list[Recipe],
) -> list[AnswerSelection]:
    """Build one AnswerSelection per verified prompt row.

    The primary text is taken from ``context_chunks.primary`` on the row.
    Sibling chunks for the other ``recipe_ids`` are loaded from the corpus
    using the chunker that matches the primary's type (per the per-type
    shapes table). ``recipe_with_a_swap`` treats its two siblings as
    reference (first non-primary) and swap target (second non-primary); all
    other multi-recipe types treat them as positional siblings.
    """
    recipes_by_id = {r.recipe_id: r for r in recipes}
    info_sections_by_recipe = {s.recipe_id: s for s in _build_info_sections(recipes)}

    selections: list[AnswerSelection] = []
    for row in prompts:
        query_type = row["query_type"]
        primary_id = int(row["_primary_id"])
        primary_recipe_id = int(row["recipe_ids"][0])
        # The prompts drafter stored the primary text under "primary" for
        # most types, but "info_chunk" for the info type. Fall back to the
        # one that's present.
        context_chunks = dict(row.get("context_chunks", {}))
        primary_text = context_chunks.get("primary") or context_chunks.get("info_chunk", "")
        recipe_ids = [int(rid) for rid in row["recipe_ids"]]
        source_prompt = row.get("source_prompt", row.get("source", "llm-draft-prompt"))

        sel = AnswerSelection(
            query=row["query"],
            query_type=query_type,
            prompt_template=row.get("prompt_template", f"{query_type}_v1"),
            primary_id=primary_id,
            primary_recipe_id=primary_recipe_id,
            primary_text=primary_text,
            recipe_ids=recipe_ids,
            source_prompt=source_prompt,
            context_chunks=context_chunks,
        )

        if query_type in TYPES_WITH_ONE_PRIMARY:
            selections.append(sel)
            continue

        other_ids = recipe_ids[1:]
        if query_type == "recipe_with_a_swap":
            # The prompts drafter wrote the swap-target text in
            # context_chunks.swap_targets (a single-element list). We
            # still need the reference text from the corpus, so load it.
            swap_targets = context_chunks.get("swap_targets") or []
            if swap_targets:
                sel.swap_text = swap_targets[0]
            # Reference is the first non-primary recipe; load from corpus
            # using the same chunker as the primary.
            sel.reference_text = _text_for_recipe(
                query_type, other_ids[0], recipes_by_id, info_sections_by_recipe,
            )
            sel.sibling_recipe_ids = other_ids
        else:
            sel.sibling_texts = [
                _text_for_recipe(
                    query_type, rid, recipes_by_id, info_sections_by_recipe,
                )
                for rid in other_ids
            ]
            sel.sibling_recipe_ids = other_ids
        selections.append(sel)
    return selections
