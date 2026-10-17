import json
from pathlib import Path
from itertools import count
from typing import Iterator
from dataclasses import dataclass
from ..corpus.preprocessor import Recipe


@dataclass
class Chunk:
    # Globally unique within a snapshot. Used as the FAISS id for dense
    # indices, so collisions across chunk sets would silently resolve a
    # hit to the wrong chunk.
    chunk_id: int
    recipe_id: int
    chunk_type: str
    text: str


@dataclass
class ChunkSet:
    name: str
    chunks: list[Chunk]


def chunk_recipes(strategy, recipes_path: Path) -> list[ChunkSet]:
    # load recipes as recipe list[Recipe]
    with open(recipes_path) as f:
        recipes = [Recipe(**recipe) for recipe in json.load(f)]

    chunk_id = count()

    if strategy == "whole_recipe":
        result = [ChunkSet(
            name="whole_recipe",
            chunks=_chunk_whole(recipes, chunk_id)
        )]

    elif strategy == "instructions_ingredients":
        result = [ChunkSet(
            name="instructions_ingredients",
            chunks=_chunk_it_ing(recipes, chunk_id)
        )]

    elif strategy == "hybrid":
        sprase_chunks = _chunk_it_ing(recipes, chunk_id)
        dense_chunks = _chunk_info(recipes, chunk_id)
        result = [
            ChunkSet(
                name="sparse",
                chunks=sprase_chunks
            ),
            ChunkSet(
                name="dense",
                chunks=dense_chunks
            )
        ]

    # Enforce the Chunk.chunk_id global-uniqueness contract. FAISS uses
    # chunk_id as the index id, so a collision would silently resolve a hit
    # to the wrong chunk in dense/retriever.py.
    all_ids = [c.chunk_id for cs in result for c in cs.chunks]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(
            "chunk_recipes produced duplicate chunk_ids - "
            "Chunk.chunk_id must be globally unique within a snapshot."
        )

    return result

    
def whole_recipe_chunks(recipes: list[Recipe]) -> list[Chunk]:
    """Public wrapper around ``_chunk_whole`` for callers outside the build pipeline.

    Allocates a fresh id stream so the returned ``Chunk.chunk_id`` values
    are unique within this call. Used by the prompts drafter, which needs
    whole-recipe chunks for nearest-neighbour selection but does not need
    the build-time ``ChunkSet`` wrapper or the global id-uniqueness contract.
    """
    return _chunk_whole(recipes, count())


def ingredient_chunks(recipes: list[Recipe]) -> list[Chunk]:
    """Public wrapper around the ingredients half of ``_chunk_it_ing``.

    Allocates a fresh id stream and returns only the chunks with
    ``chunk_type == "ingredients"``. Used by the prompts drafter.
    """
    return [c for c in _chunk_it_ing(recipes, count()) if c.chunk_type == "ingredients"]


def instruction_chunks(recipes: list[Recipe]) -> list[Chunk]:
    """Public wrapper around the instructions half of ``_chunk_it_ing``.

    Allocates a fresh id stream and returns only the chunks with
    ``chunk_type == "instructions"``. Used by the prompts drafter.
    """
    return [c for c in _chunk_it_ing(recipes, count()) if c.chunk_type == "instructions"]


def _chunk_whole(recipes: list[Recipe], chunk_id: Iterator[int]) -> list[Chunk]:
    """Produce one chunk per recipe holding the full recipe rendered as a single block of text.
    Used by the dense strategy, where each recipe is a single retrieval unit.
    """
    chunks = []
    for r in recipes:
        chunks.append(
            Chunk(
                text=_format_whole_recipe(r),
                chunk_type="whole",
                recipe_id=r.recipe_id,
                chunk_id=next(chunk_id)
            )
        )
    return chunks


def _chunk_it_ing(recipes: list[Recipe], chunk_id: Iterator[int]) -> list[Chunk]:
    """Produce one ingredients chunk and one instructions chunk per recipe.
    Used by the sparse strategy and as the sparse half of the hybrid strategy, where ingredients and instructions are tokenised separately for BM25.
    """
    chunks = []
    for r in recipes:
        chunks.append(
            Chunk(
                chunk_id=next(chunk_id),
                recipe_id=r.recipe_id,
                chunk_type="ingredients",
                text=_format_ingredients(r)
            )
        )
        chunks.append(
            Chunk(
                chunk_id=next(chunk_id),
                recipe_id=r.recipe_id,
                chunk_type="instructions",
                text=_format_instructions(r)
            )
        )
    return chunks


def _chunk_info(recipes: list[Recipe], chunk_id: Iterator[int]) -> list[Chunk]:
    """Produce one chunk per entry in each recipe's info section, so an entry can be retrieved on its own.
    Used as the dense half of the hybrid strategy, where the entry-level granularity gives the cross-encoder finer rerank signal than a joined section would.
    """
    chunks = []
    for r in recipes:
        for i in r.info:
            chunks.append(
                Chunk(
                    chunk_id=next(chunk_id),
                    recipe_id=r.recipe_id,
                    chunk_type="info",
                    text=_format_info(i)
                )
        )
    return chunks


def _format_ingredients(recipe: Recipe) -> str:
    """Render a recipe's ingredient groups into a markdown-ish block. 
    Each group gets an `##` heading when it has a name; each ingredient becomes a `- amount unit name (notes) or alt_amount alt_unit` line. 
    The trailing blank line per group keeps the rendered text readable when groups are concatenated.
    """
    lines = []

    for group in recipe.ingredient_groups:
        # assign group name if exists
        group_name = group.get("group_name")
        if group_name:
            lines.append(f"## {group_name}")

        for ing in group.get("ingredients", []):
            parts = []

            if ing.get("amount"):
                parts.append(str(ing["amount"]))
            if ing.get("unit"):
                parts.append(ing["unit"])

            parts.append(ing.get("name", "Unknown"))

            line = " ".join(parts)

            if ing.get("notes"):
                line += f" ({ing['notes']})"

            if ing.get("alt_amount"):
                line += f" or {ing['alt_amount']} {ing['alt_unit']}"

            lines.append(f"- {line}")

        lines.append("")

    return "\n".join(lines).strip()


def _format_instructions(recipe: Recipe) -> str:
    """Render a recipe's instruction groups into a single block. 
    Group headings are emitted only when the group has a name; steps are written as plain lines, 
    preserving whatever numbering the source already carries (no renumbering is applied here, unlike `_format_whole_recipe`).
    """
    lines = []

    for group in recipe.instruction_groups:
        # assign group name if exists
        group_name = group.get("group_name")
        if group_name:
            lines.append(f"## {group_name}")

        for it in group.get("instructions", []):
            lines.append(f"{it}")

    return "\n".join(lines)


def _format_info(info: dict) -> str:
    """Render a single info entry as a `## group_name` heading followed by the entry content, or just the content when no group name is set. 
    This is the per-entry shape the hybrid dense index embeds.
    """
    lines = []

    group_name = info.get("group_name")
    if group_name:
        lines.append(f"## {group_name}")
    lines.append(f"{info.get("content", "")}")

    return "\n".join(lines)


def _format_whole_recipe(recipe: Recipe) -> str:
    """Render a recipe as a single block of text suitable for dense embedding: a header line with the name and servings, 
    then the ingredient groups, then numbered instructions, then optional notes. 
    Instruction group headings named `Full Recipe`, `Abbreviated Recipe`, `Abbreviated`, or `Main` are suppressed, 
    since those labels carry no semantic content and would only add noise to the embedding.
    """
    lines = []

    # header
    lines.append(f"Recipe Name: {recipe.recipe_name}")

    # sevings
    servings = recipe.servings
    if servings:
        lines.append(f"Servings: {servings}")

    # ingredient groups
    lines.append("\n## Ingredients")
    for ing_group in recipe.ingredient_groups:
        lines.append(f"\n### {ing_group.get('group_name')}")
        for ing in ing_group.get("ingredients", []):
            parts = []
            if ing.get("amount"):
                parts.append(str(ing["amount"]))
            if ing.get("unit"):
                parts.append(ing["unit"])
            parts.append(ing.get("name", "Unknown"))
            
            line = " ".join(parts)

            if ing.get("notes"):
                line += f" ({ing['notes']})"
            if ing.get("alt_amount"):
                line += f" or ({ing['alt_amount']}) {ing['alt_unit']}" # of alt amount, then alt unit exists
            
            lines.append(f" - {line}")
        
    # instruction groups
    lines.append("\n## Instructions")
    for group in recipe.instruction_groups:
        name = group.get("group_name", "")
        if name not in ("Full Recipe", "Abbreviated Recipe", "Abbreviated", "Main"):
            lines.append(f"\n### {name}")
        for i, step in enumerate(group.get("instructions", []), 1):
            lines.append(f"{i}. {step}")

    # notes
    notes = recipe.notes
    if notes:
        lines.append("\n## Notes")
        for note in notes:
            lines.append(f"- {note}")
    
    return "\n".join(lines)