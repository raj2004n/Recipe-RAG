from dataclasses import replace

from src.corpus.preprocessor import Recipe
from src.domain.source_block import source_block, user_message


def _ingredient(**fields) -> dict:
    """One ingredient as the corpus stores it, with every key present."""
    base = {"amount": "", "unit": "", "name": "", "notes": "", "alt_amount": "", "alt_unit": ""}
    return base | fields


SAUCE = Recipe(
    recipe_id=1,
    recipe_name="Test Taco Sauce",
    servings=4.0,
    ingredient_groups=[
        {
            "group_name": "Main",
            "ingredients": [
                _ingredient(amount="1/2", unit="cup", name="sour cream"),
                _ingredient(notes="optional"),
            ],
        },
        {
            "group_name": "Extras",
            "ingredients": [_ingredient(amount="1", unit="tsp", name="salt")],
        },
    ],
    instruction_groups=[
        {
            "group_name": "Main",
            "instructions": ["Mix everything until smooth.", "Taste and adjust."],
        },
    ],
)

EXPECTED_BLOCK = "\n".join(
    [
        "RECIPE 1",
        "Title: Test Taco Sauce",
        "Servings: 4",
        "Ingredients:",
        "## Main",
        "- 1/2 cup sour cream",
        "-  (optional)",
        "## Extras",
        "- 1 tsp salt",
        "Instructions:",
        "## Main",
        "Mix everything until smooth.",
        "Taste and adjust.",
    ]
)


def test_source_block_renders_the_four_fields_in_order():
    assert source_block("RECIPE 1", SAUCE) == EXPECTED_BLOCK


def test_servings_lose_their_float():
    block = source_block("RECIPE 1", SAUCE)
    assert "Servings: 4" in block
    assert "4.0" not in block


def test_a_fractional_servings_count_survives():
    block = source_block("RECIPE 1", replace(SAUCE, servings=2.5))
    assert "Servings: 2.5" in block


def test_a_recipe_without_servings_drops_the_line():
    block = source_block("RECIPE 1", replace(SAUCE, servings=0.0))
    assert "Servings:" not in block
    assert block.split("\n")[:3] == ["RECIPE 1", "Title: Test Taco Sauce", "Ingredients:"]


def test_blank_lines_go_but_noisy_lines_stay():
    block = source_block("RECIPE 1", SAUCE)
    assert "\n\n" not in block
    assert "-  (optional)" in block


def test_steps_are_not_renumbered():
    assert "1. Mix everything until smooth." not in source_block("RECIPE 1", SAUCE)


def test_user_message_puts_both_recipes_before_the_request():
    message = user_message(SAUCE, SAUCE, "Invent a pie.")
    assert message.startswith("RECIPE 1\nTitle: Test Taco Sauce")
    assert "\n\nRECIPE 2\nTitle: Test Taco Sauce" in message
    assert message.endswith("\n\nInvent a pie.")


def test_the_request_is_the_last_thing_in_the_message():
    message = user_message(SAUCE, SAUCE, "Invent a pie.")
    assert message.index("Invent a pie.") > message.index("RECIPE 2")
