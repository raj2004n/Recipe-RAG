import json
import pprint
from itertools import count
from src.rag.core.snapshot import Chunk
from src.rag.data.preprocessor import Recipe
from dataclasses import dataclass

# TODO: Notes saved twice. Notes is not scanned for retrieval so can be discared, then same for serving.

# add chunk id

# TODO: Chunks by notes? Notes should be considered earlier. Currently, only ing and it are independently, and only ing notes are considered.

def format_ingredients(recipe: dict) -> str:
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

def format_instructions(recipe: dict) -> str:
    lines = []
    
    for group in recipe.get("instruction_groups", []):
        # assign group name if exists
        group_name = group.get("group_name")
        if group_name:
            lines.append(f"## {group_name}")

        for it in group.get("instructions", []):
            lines.append(f"{it}")
    
    return "\n".join(lines)

def format_instruction_group(group: dict) -> str:
    lines = []

    name = group.get("group_name")
    if name:
        lines.append(f"## {name}")

    for i, step in enumerate(group.get("instructions", []), 1):
        lines.append(f"{i}. {step}")

    return "\n".join(lines)

"""  
How to handle this:
Two chunking:
Dense retriever:
- recipe_id
- chunk_id
- info

Sparse retriever
- recipe_id
- chunk_id
- ing
- it
- notes
- servings

Both are different sets, their chunk ids will not match.

Should have a new dataclass and rename the other
Chunk -> SparseChunk
_____ -+-> DenseChunk
"""

# TODO: Need to see what chunks (size, contents, etc) a sparse retriever prefers
# consider if i need to split the chunks up by types

#TODO: How does uneven sized chunks impact dense retrieval / sparse retrieval

# TODO how much are these headers helping for dense chunks? e.g. ## General info\n ...content...

def chunk_info(recipes: list[Recipe]):
    # for dense retriever
    chunks = []
    chunk_id = count()

    for r in recipes:
        info = r.info
        for i in info:
            chunks.append(
                Chunk(
                    chunk_id=next(chunk_id),
                    recipe_id=r.recipe_id,
                    chunk_type="info",
                    text=_format_info(i)
                )
        )
    
    [print(f"{c}\n\n") for c in chunks]
    return


def chunk_it_ing(recipes: list[Recipe]) -> list[Chunk]:
    """ Chunk by instructions and ingredients"""
    chunks = []
    chunk_id = count()

    for r in recipes:
        chunks.append(
            Chunk(
                chunk_id=next(chunk_id),
                recipe_id=r.recipe_id,
                chunk_type="ingredients",
                text=format_ingredients(r)
            )
        )
        chunks.append(
            Chunk(
                chunk_id=next(chunk_id),
                recipe_id=r.recipe_id,
                chunk_type="instructions",
                text=format_instructions(r)
            )
        )
    return chunks


def _format_info(info: dict) -> str:
    lines = []
    
    group_name = info.get("group_name")
    if group_name:
        lines.append(f"## {group_name}")
    lines.append(f"{info.get("content", "")}")
    
    return "\n".join(lines)

def test():
    # load processed recipes
    recipes_path = "data/processed/cleaned_recipes.json"

    # load as Recipe dataclass
    with open(recipes_path) as f:
        recipes = [Recipe(**recipe) for recipe in json.load(f)]
    
    chunk_info(recipes[:2])

if __name__ == "__main__":
    test()