def recipe_to_string(recipe: dict) -> str:
    """Flatten a recipe dict into a clean, searchable text block.

    Consumes the raw recipe dict (as returned by the JSON corpus or the
    preprocessor's `asdict`) and emits a single markdown-ish string with
    sections for the name, ingredients, instructions, optional notes, and
    servings. Used by the hybrid pipeline: after rerank, chunks are
    resolved back to whole recipe dicts and this helper renders them into
    the prompt context that `PromptBuilder` token-budgets before sending to
    the generator.

    The shape overlaps with the chunker's `_format_whole_recipe` but is
    deliberately simpler: instructions are listed as bullets rather than
    re-numbered, ingredient lines drop the alt-amount/alt-unit branch, and
    the section ordering (name, ingredients, instructions, notes, servings)
    is fixed so rerank hits read consistently regardless of which recipe
    produced them.
    """
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
