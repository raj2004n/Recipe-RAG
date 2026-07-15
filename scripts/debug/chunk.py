import json
import pprint
from itertools import count

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

def format_instructions(recipe: dict):
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

def chunk_it_ing(recipes: list[dict]):
    """ Chunk by instructions and ingredients"""
    chunks = []
    chunk_id = count()

    for r in recipes:
        recipe_id = r.get("recipe_id")

        # ingredients
        chunks.append({
            "text": format_ingredients(r),
            "chunk_type": "ingredients",
            "recipe_id": recipe_id,
            "chunk_id": next(chunk_id),
        })

        # instructions
        chunks.append({
            "text": format_instructions(r),
            "chunk_type": "instructions",
            "recipe_id": recipe_id,
            "chunk_id": next(chunk_id)
        })
    
    return chunks

def test():
    # load processed recipes
    recipes_path = "data/processed/cleaned_recipes.json"
    with open(recipes_path) as f:
        recipes = json.load(f)

    chunks = chunk_it_ing(recipes[:5])

    #print(f"="*50)
    #groups = [g for g in r.get("instruction_groups", [])]
    #print(f"Groups: {groups}")
    
    #print(f"="*50)
    #print(f"-----Formatted instructions groups-----")
    #for g in groups:
        #print(format_instruction_group(g))

    #print(f"="*50)
    #print(f"-----Formatted instructions (whole)-----")
    #print(format_instructions(r))

    #print(f"="*50)
    #print(f"-----Formatted ingredients (whole)-----")
    #print(format_ingredients(r))

    print(f"="*50)
    print(f"-----Chunks-----")
    pprint.pprint(chunks)

if __name__ == "__main__":
    test()