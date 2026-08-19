"""
Format whole recipes into a single text block for prompts / reranking.
"""


def recipe_to_string(recipe: dict) -> str:
    """Flatten a recipe dict into a clean, searchable text block"""
    parts: list[str] = [f"Recipe Name: {recipe.get('recipe_name', '')}"]

    parts.append("Ingredients:")
    for group in recipe.get("ingredient_groups", []):
        for ing in group.get("ingredients", []):
            amount = " ".join(
                str(ing.get(k, "") or "")
                for k in ("amount", "unit")
            ).strip()
            notes = f" ({ing.get('notes')})" if ing.get("notes") else ""
            name = ing.get("name", "Unknown")
            parts.append(f"- {amount} {name}{notes}".strip())

    parts.append("Instructions:")
    for group in recipe.get("instruction_groups", []):
        for step in group.get("instructions", []):
            parts.append(f"- {step}")

    recipe_notes = recipe.get("notes") or []
    if recipe_notes:
        parts.append("Notes:")
        for note in recipe_notes:
            parts.append(f"- {note}")

    if recipe.get("servings"):
        parts.append(f"Servings: {recipe['servings']}")

    return "\n".join(parts)
