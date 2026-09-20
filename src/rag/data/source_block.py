"""Render retrieved recipes into the source block the teacher is handed.

A training row shows a teacher model two recipes and the request they were chosen for, and asks it
to invent one new dish out of them. This module renders the first half of that row: a recipe's
label, its name, its servings, and the two working sections.

Ingredients and steps come from the chunker's own formatters, so the teacher reads the same text
the sparse index is built from, and the same text the pair prompt was designed against. Everything
inside those two sections is copied exactly as the formatter returns it, apart from blank lines,
which are dropped. The corpus carries the usual scraping noise, a line holding nothing but a note
among them, and a block is a faithful copy of a page rather than a tidy one. Steps are never
renumbered: they stay in whatever shape the source wrote them.
"""

from src.rag.data.preprocessor import Recipe
from src.rag.retrieval.chunker import ingredient_chunks, instruction_chunks


def source_block(label: str, recipe: Recipe) -> str:
    """One recipe as the teacher reads it: the label, then the four fields in a fixed order.

    The servings line is left out when the recipe carries none, which is the only way a block
    comes out with three fields instead of four.
    """
    lines = [label, f"Title: {recipe.recipe_name}"]

    if recipe.servings:
        lines.append(f"Servings: {_servings_text(recipe.servings)}")

    lines.append("Ingredients:")
    lines.extend(_content_lines(_ingredients_text(recipe)))
    lines.append("Instructions:")
    lines.extend(_content_lines(_instructions_text(recipe)))

    return "\n".join(lines)


def user_message(primary: Recipe, partner: Recipe, request: str) -> str:
    """The teacher's user turn: both recipes, then the request they were chosen for.

    The recipes come first because a request refers back to them by position, and the labels are
    what make "the first recipe" unambiguous. The request goes in untouched, since it is the user
    turn the student is being trained to answer.
    """
    return (
        f"{source_block('RECIPE 1', primary)}\n\n"
        f"{source_block('RECIPE 2', partner)}\n\n"
        f"{request}"
    )


def _servings_text(servings: float) -> str:
    """The corpus stores servings as a float, so four servings would otherwise read as "4.0"."""
    value = float(servings)
    return str(int(value)) if value.is_integer() else str(value)


def _ingredients_text(recipe: Recipe) -> str:
    """A recipe's ingredients, through the formatter the sparse index is built from."""
    return ingredient_chunks([recipe])[0].text


def _instructions_text(recipe: Recipe) -> str:
    """A recipe's instructions, through the formatter the sparse index is built from."""
    return instruction_chunks([recipe])[0].text


def _content_lines(text: str) -> list[str]:
    """The formatter's output with its blank lines dropped, so a section reads as one run."""
    return [line for line in text.split("\n") if line.strip()]
