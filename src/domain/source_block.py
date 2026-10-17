from src.corpus.preprocessor import Recipe
from src.domain.chunker import ingredient_chunks, instruction_chunks


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


def labelled_blocks(recipes: list[Recipe]) -> list[str]:
    """One block per recipe, each labelled by its position in the turn it will be read in."""
    return [
        source_block(f"RECIPE {index}", recipe)
        for index, recipe in enumerate(recipes, start=1)
    ]


def request_message(blocks: list[str], request: str) -> str:
    """A user turn assembled from rendered blocks: each one in order, then the request.

    The recipes come first because a request refers back to them by position, and the labels are
    what make "the first recipe" unambiguous. The request goes in untouched, since it is the user
    turn the model is being asked to answer. A turn is laid out by how many blocks it was handed and
    nothing else, so a pair of training sources and a longer serving context come out of one code
    path, and a caller that drops a block drops it from the end.
    """
    return "\n\n".join([*blocks, request])


def user_message(primary: Recipe, partner: Recipe, request: str) -> str:
    """The teacher's user turn: both recipes, then the request they were chosen for."""
    return request_message(labelled_blocks([primary, partner]), request)


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
