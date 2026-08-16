import json
from pathlib import Path
from itertools import count
from typing import Iterator
from dataclasses import dataclass
from ..data.preprocessor import Recipe

# TODO: Gracefully handle when recipes not found for loading

@dataclass
class Chunk:
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
    
    if strategy == "whole_recipe":
        return [ChunkSet(
            name="whole_recipe",
            chunks=_chunk_whole(recipes)
        )]
    
    elif strategy == "instructions_ingredients":
        return [ChunkSet(
            name="instructions_ingredients",
            chunks=_chunk_it_ing(recipes)
        )]
    
    elif strategy == "hybrid":
        sprase_chunks = _chunk_it_ing(recipes)
        dense_chunks = _chunk_info(recipes)
        return [
            ChunkSet(
                name="sparse",
                chunks=sprase_chunks
            ),
            ChunkSet(
                name="dense",
                chunks=dense_chunks
            )
        ]

    
def _chunk_whole(recipes: list[Recipe], chunk_id: Iterator[int]) -> list[Chunk]:
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
    lines = []
    
    group_name = info.get("group_name")
    if group_name:
        lines.append(f"## {group_name}")
    lines.append(f"{info.get("content", "")}")
    
    return "\n".join(lines)


def _format_whole_recipe(recipe: Recipe) -> str:
    """ Format recipe text for embedding. """
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