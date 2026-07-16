import json
from pathlib import Path
from itertools import count
from dataclasses import dataclass

# TODO: Gracefully handle when recipes not found for loading

@dataclass
class Chunk:
    text: str
    chunk_type: str
    recipe_id: int
    chunk_id: int

def chunk_recipes(strategy, recipes_path: Path) -> list[Chunk]:
    # load recipes
    with open(recipes_path) as f:
        recipes = json.load(f)
    
    if strategy == "whole_recipe":
        return _chunk_whole(recipes)
    
    elif strategy == "instructions_ingredients":
        return _chunk_it_ing(recipes)
    
def _chunk_whole(recipes: list[dict]) -> list[Chunk]:
    chunks = []
    for r in recipes:
        recipe_id = r.get("recipe_id")
        chunk_id = count()

        chunks.append(
            Chunk(
                text=_format_whole_recipe(r),
                chunk_type="whole",
                recipe_id=recipe_id,
                chunk_id=next(chunk_id)
            )
        )
    
    return chunks

def _chunk_it_ing(recipes: list[dict]) -> list[Chunk]:
    chunks = []

    for r in recipes:
        recipe_id = r.get("recipe_id")
        chunk_id = count()

        # ingredients
        chunks.append(
            Chunk(
                text=_format_ingredients(r),
                chunk_type="ingredients",
                recipe_id=recipe_id,
                chunk_id=next(chunk_id)
            )
        )

        # instructions
        chunks.append(
            Chunk(
                text=_format_instructions(r),
                chunk_type="instructions",
                recipe_id=recipe_id,
                chunk_id=next(chunk_id)
            )
        )
    
    return chunks

def _format_ingredients(recipe: dict) -> str:
    lines = []

    for group in recipe.get("ingredient_groups", []):
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

def _format_instructions(recipe: dict) -> str:
    lines = []

    for group in recipe.get("instruction_groups", []):
        # assign group name if exists
        group_name = group.get("group_name")
        if group_name:
            lines.append(f"## {group_name}")

        for it in group.get("instructions", []):
            lines.append(f"{it}")
    
    return "\n".join(lines)

def _format_whole_recipe(recipe: dict) -> str:
    """ Format recipe text for embedding. """
    lines = []

    # header
    lines.append(f"Recipe Name: {recipe.get('recipe_name')}")

    # sevings
    servings = recipe.get("servings", 1)
    if servings:
        lines.append(f"Servings: {servings}")

    # ingredient groups
    lines.append("\n## Ingredients")
    for ing_group in recipe.get("ingredient_groups", []):
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
    for group in recipe.get("instruction_groups", []):
        name = group.get("group_name", "")
        if name not in ("Full Recipe", "Abbreviated Recipe", "Abbreviated", "Main"):
            lines.append(f"\n### {name}")
        for i, step in enumerate(group.get("instructions", []), 1):
            lines.append(f"{i}. {step}")

    # notes
    notes = recipe.get("notes", [])
    if notes:
        lines.append("\n## Notes")
        for note in notes:
            lines.append(f"- {note}")
    
    return "\n".join(lines)