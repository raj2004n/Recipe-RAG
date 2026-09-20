"""Tests for the parser that turns a generated answer into the sections the
result view draws.

Half of these are unit tests over hand-written strings, one per rule the
parser applies. The other half run over eight answers recorded from the
served model, because every rule here is a guess about what a 1B model
writes, and the recording is the only thing that says whether the guesses
hold.
"""

import json
import re
from pathlib import Path

import pytest

from src.rag.serve.parser import (
    META_LABELS,
    NON_INGREDIENT_LABELS,
    SECTION_OPENERS,
    Ingredient,
    IngredientGroup,
    MetaFact,
    StepGroup,
    parse_answer,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "generator_outputs"
RECORDINGS = FIXTURES / "outputs.jsonl"
SIDECAR = FIXTURES / "sidecar.json"

# The words of the labels a parse lifts a line out for: the section openers,
# the name of a block that is not an ingredient list, the labels stating what
# a recipe yields, and the two words an answer names the dish with. They are
# labels rather than content, so a line whose words are missing because a
# label was read off it is not a lost line.
LABEL_WORDS = (
    {"recipe", "name"}
    | {
        word
        for openers in SECTION_OPENERS.values()
        for opener in openers
        for word in opener.split()
    }
    | {word for label in NON_INGREDIENT_LABELS for word in label.split()}
    | {word for label in META_LABELS for word in label.split()}
)

WORD = re.compile(r"[a-z0-9']{4,}")

# The name each recorded answer gives its dish. The model names it three
# different ways across the eight: a markdown heading, a "Recipe Name:" label
# under a sentence of preamble, and a bold span inside the opening sentence.
TITLES = {
    "plain-dinner": "Speedy Lemon Chicken & Veggie Sheet Pan Dinner",
    "comforting": "Spiced Apple & Maple Crumble",
    "exact-ingredients": (
        "Honey Soy Glazed Chicken Thighs with Ginger-Garlic Rice"
    ),
    "vegetarian-pasta": "Sun-Kissed Tomato & Spinach Pasta Primavera",
    "asks-for-tips": "The Golden Glory Roast Chicken: A Step-by-Step Guide",
    "two-components": "Spicy Korean Pork Belly & Kimchi Fried Rice",
    "prose-question": (
        "Golden Brilliance: Crispy Onion Frying – A Recipe for Perfection"
    ),
    "long-answer": "Harvest Bounty: A Three-Course Dinner for Six",
}


@pytest.fixture(scope="module")
def recorded():
    """The recorded answers by id, as the served model wrote them."""
    return {
        json.loads(line)["id"]: json.loads(line)["answer"]
        for line in RECORDINGS.read_text().splitlines()
        if line
    }


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def parsed_text(parsed: dict) -> str:
    """Every string a parse produced, as one block of text."""
    parts = [parsed["title"]]
    parts += [
        f"{fact['label']} {fact['value']}" for fact in parsed["meta"]
    ]
    for entry in parsed["ingredients"]:
        if entry["kind"] == "group":
            parts.append(entry["heading"])
        else:
            parts.append(f"{entry['quantity']} {entry['text']}")
    for step in parsed["steps"]:
        parts.append(step["heading"] if step["kind"] == "group" else step["text"])
    parts += parsed["notes"]
    parts.append(parsed["remainder"])
    return " ".join(parts)


# The rules, one test each.


def test_a_heading_is_the_title():
    source = "## Lemon Chicken\n\n**Ingredients:**\n*   1 lemon\n"
    assert parse_answer(source).title == "Lemon Chicken"


def test_a_named_recipe_under_a_sentence_of_preamble_is_the_title():
    """The model often says what it is about to do before it names the dish.
    The preamble is kept as prose rather than read as the title."""
    source = (
        "Here is a lovely recipe for you, and one that is easy to follow:\n"
        "\n"
        "**Recipe Name:** Lemon Chicken\n"
        "\n"
        "**Ingredients:**\n"
        "*   1 lemon\n"
    )
    answer = parse_answer(source)
    assert answer.title == "Lemon Chicken"
    assert answer.remainder == (
        "Here is a lovely recipe for you, and one that is easy to follow:"
    )


def test_a_bold_name_inside_the_opening_sentence_is_the_title():
    source = (
        "The recipe is: **Lemon Chicken**\n\n**Ingredients:**\n*   1 lemon\n"
    )
    assert parse_answer(source).title == "Lemon Chicken"


def test_prose_that_comes_first_is_not_a_title():
    source = (
        "Preheat the oven and then decide what you are making this evening.\n"
        "\n"
        "**Ingredients:**\n*   1 lemon\n"
    )
    assert parse_answer(source).title == "1 lemon"


def test_the_labels_over_the_recipe_become_facts():
    """What a recipe makes and how long it takes are drawn as badges rather
    than left in the prose, because a cook reads them first."""
    source = (
        "**Yields:** 4 servings\n"
        "**Prep time:** 15 minutes\n"
        "**Cook time:** 20 minutes\n"
        "\n"
        "**Ingredients:**\n*   1 lemon\n"
    )
    assert parse_answer(source).meta == (
        MetaFact(label="Yields", value="4 servings"),
        MetaFact(label="Prep time", value="15 minutes"),
        MetaFact(label="Cook time", value="20 minutes"),
    )


def test_the_same_fact_twice_is_one_badge_and_two_values_are_two():
    """An answer that writes three recipes states the same yields three
    times. The repeat says nothing and is dropped; 15 minutes and 25 minutes
    are different facts and both are kept."""
    source = (
        "**Yields:** 4 servings\n"
        "**Yields:** 4 servings\n"
        "**Prep time:** 15 minutes\n"
        "**Prep time:** 25 minutes\n"
        "\n"
        "**Ingredients:**\n*   1 lemon\n"
    )
    assert [fact.value for fact in parse_answer(source).meta] == [
        "4 servings",
        "15 minutes",
        "25 minutes",
    ]


def test_a_step_named_with_a_word_a_badge_uses_is_still_a_step():
    """The badge labels are matched whole. A step the model titled "Chill:"
    is a step, not a fact about the recipe."""
    source = (
        "**Ingredients:**\n*   1 lemon\n"
        "\n"
        "**Instructions:**\n"
        "1. **Chill:** Put it in the refrigerator for an hour.\n"
    )
    answer = parse_answer(source)
    assert answer.meta == ()
    assert answer.steps == ("Chill: Put it in the refrigerator for an hour.",)


def test_a_label_over_its_own_items_heads_a_group():
    source = (
        "## Crumble\n\n"
        "**Ingredients:**\n"
        "*   **For the Filling:**\n"
        "    *   2 apples\n"
        "    *   1 tbsp sugar\n"
        "*   100g flour\n"
    )
    assert parse_answer(source).ingredients == (
        IngredientGroup(heading="For the Filling"),
        Ingredient(quantity="2", text="apples"),
        Ingredient(quantity="1 tbsp", text="sugar"),
        Ingredient(quantity="100g", text="flour"),
    )


def test_a_label_with_its_ingredient_on_the_same_line_stays_an_ingredient():
    """A bold label that carries nothing under it is part of the ingredient,
    not a heading over a group, even though it looks like one."""
    source = (
        "**Ingredients:**\n"
        "*   **Chicken:** 6-8 boneless thighs\n"
        "*   **Vegetables:** 1 lb broccoli\n"
    )
    assert parse_answer(source).ingredients == (
        Ingredient(quantity="", text="Chicken: 6-8 boneless thighs"),
        Ingredient(quantity="", text="Vegetables: 1 lb broccoli"),
    )


def test_an_unlabelled_line_over_a_list_heads_a_group():
    source = (
        "**Ingredients:**\n"
        "For the skillet:\n"
        "*   2 tbsp oil\n"
        "*   1 onion\n"
        "\n"
        "**Instructions:**\n1. Fry it.\n"
    )
    assert parse_answer(source).ingredients == (
        IngredientGroup(heading="For the skillet"),
        Ingredient(quantity="2 tbsp", text="oil"),
        Ingredient(quantity="1", text="onion"),
    )


def test_equipment_ends_the_ingredient_block_and_is_kept_as_prose():
    """A roasting pan is not something to cook with, so the list of kit is
    kept out of the ingredient rows and shown with the rest of the prose."""
    source = (
        "**Ingredients:**\n"
        "*   1 chicken\n"
        "\n"
        "**Equipment:**\n"
        "*   Roasting pan with a rack\n"
        "*   Tongs\n"
        "\n"
        "**Instructions:**\n"
        "1. Roast it.\n"
    )
    answer = parse_answer(source)
    assert answer.ingredients == (Ingredient(quantity="1", text="chicken"),)
    assert answer.steps == ("Roast it.",)
    assert answer.remainder == "Equipment:\nRoasting pan with a rack\nTongs"


def test_a_numbered_stage_heads_the_steps_it_covers():
    """Numbering a stage of the work is how the model groups steps. The
    stage holds the timing, so it is drawn as a heading over those steps
    rather than numbered as one of them."""
    source = (
        "**Ingredients:**\n*   1 chicken\n"
        "\n"
        "**Instructions:**\n"
        "**1. Prepare the Chicken (30 minutes)**\n"
        "*   Bring it to room temperature.\n"
        "*   Pat it dry.\n"
        "**2. Roast it (1 hour)**\n"
        "*   Roast until done.\n"
    )
    assert parse_answer(source).steps == (
        StepGroup(heading="Prepare the Chicken (30 minutes)"),
        "Bring it to room temperature.",
        "Pat it dry.",
        StepGroup(heading="Roast it (1 hour)"),
        "Roast until done.",
    )


def test_an_unlabelled_stage_over_a_paragraph_is_not_a_heading():
    """A long bold sentence over prose is not a name for a group of items.
    It is kept as prose, which is where the model's own course headings
    belong."""
    source = (
        "## Dinner\n\n"
        "**Ingredients:**\n*   1 lemon\n"
        "\n"
        "**Instructions:**\n"
        "1. Cook it.\n"
        "\n"
        "**Course 2: A Long Roasted Vegetable Medley With An Herb Crust**\n"
        "\n"
        "**Ingredients:**\n*   2 carrots\n"
    )
    answer = parse_answer(source)
    assert [step for step in answer.steps if isinstance(step, StepGroup)] == []
    assert answer.remainder == (
        "Course 2: A Long Roasted Vegetable Medley With An Herb Crust"
    )
    assert answer.ingredients == (
        Ingredient(quantity="1", text="lemon"),
        Ingredient(quantity="2", text="carrots"),
    )


def test_step_numbering_is_the_screens_own():
    """Whatever the model numbered a step with is dropped. The screen numbers
    the list itself, so its 1 to n is never a copy of the model's."""
    source = (
        "**Instructions:**\n"
        "1.  First thing.\n"
        "2) Second thing.\n"
        "*   Third thing.\n"
    )
    assert parse_answer(source).steps == (
        "First thing.",
        "Second thing.",
        "Third thing.",
    )


@pytest.mark.parametrize(
    "opener",
    [
        "Notes:",
        "Notes & Tips:",
        "Tips & Notes:",
        "Notes and Tips:",
        "Tips for Extra Crispness:",
    ],
)
def test_every_notes_opener_the_model_reaches_for_opens_the_notes(opener):
    source = f"**Ingredients:**\n*   1 lemon\n\n**{opener}**\n*   Keep it cool.\n"
    assert parse_answer(source).notes == ("Keep it cool.",)


def test_the_marks_the_model_emphasised_with_are_taken_off():
    """The page renders text and never markup, so an asterisk that survived
    would be read as one."""
    source = (
        "**Ingredients:**\n"
        "*   2 tbsp **olive oil**\n"
        "\n"
        "**Instructions:**\n"
        "1. **Warm** it *gently*.\n"
        "\n"
        "**Notes:**\n"
        "*   *Do not* overheat it.\n"
    )
    answer = parse_answer(source)
    assert answer.ingredients == (
        Ingredient(quantity="2 tbsp", text="olive oil"),
    )
    assert answer.steps == ("Warm it gently.",)
    assert answer.notes == ("Do not overheat it.",)


def test_an_amount_written_without_a_space_is_still_an_amount():
    """The corpus writes "100g butter" as often as "100 g butter", and either
    way the amount belongs in its own column."""
    source = (
        "**Ingredients:**\n"
        "*   100g butter\n"
        "*   500ml milk\n"
        "*   1.5 kg potatoes\n"
        "*   1 1/2 cups rice\n"
    )
    assert [entry.quantity for entry in parse_answer(source).ingredients] == [
        "100g",
        "500ml",
        "1.5 kg",
        "1 1/2 cups",
    ]


def test_an_answer_with_no_sections_comes_back_whole():
    """An answer written as prose is not a recipe view. It is returned as it
    was written, with no title and nothing laid out, and the page is told so,
    which is what makes it fall back to showing the text rather than an empty
    page under a heading."""
    source = "Butter browns faster than oil because of the milk solids."
    answer = parse_answer(source)
    assert not answer.parsed
    assert answer.as_dict() == {
        "title": "",
        "meta": [],
        "ingredients": [],
        "steps": [],
        "notes": [],
        "remainder": source,
        "parsed": False,
    }


def test_the_page_is_told_a_group_from_an_item():
    source = (
        "**Ingredients:**\n"
        "*   **For the sauce:**\n"
        "    *   1 tbsp soy sauce\n"
        "*   1 lime\n"
    )
    parsed = parse_answer(source).as_dict()
    assert parsed["parsed"] is True
    assert parsed["ingredients"] == [
        {"kind": "group", "heading": "For the sauce"},
        {"kind": "item", "quantity": "1 tbsp", "text": "soy sauce"},
        {"kind": "item", "quantity": "1", "text": "lime"},
    ]


# The recorded answers, which are what the rules above were fitted to.


@pytest.mark.parametrize("name", sorted(TITLES))
def test_each_recorded_answer_is_titled_from_its_own_name(recorded, name):
    assert parse_answer(recorded[name]).title == TITLES[name]


@pytest.mark.parametrize("name", sorted(TITLES))
def test_no_long_word_of_a_recorded_answer_is_dropped(recorded, name):
    """Every ingredient, step, note and aside in the recording is drawn from
    one of the parsed fields. The only words a parse may lose are the ones it
    read as a label, so a section that stopped being parsed shows up here as
    a pile of missing words."""
    answer = recorded[name]
    missing = (
        words(answer)
        - words(parsed_text(parse_answer(answer).as_dict()))
        - LABEL_WORDS
    )
    assert missing == set()


@pytest.mark.parametrize("name", sorted(TITLES))
def test_every_recorded_answer_arrives_as_a_recipe(recorded, name):
    """The generation prompt asks for a recipe and says nothing about the
    headings, so this is the plan's high risk measured: none of the eight
    answers came back as prose with nothing to lay out."""
    parsed = parse_answer(recorded[name]).as_dict()
    assert parsed["ingredients"]
    assert parsed["steps"]
    assert parsed["meta"]


def test_the_recorded_answers_group_their_ingredients(recorded):
    """Six of the eight put their ingredients under a bold label with the
    ingredients as bullets one level in. The label names what the group is
    for, so it is a heading over those ingredients."""
    parsed = parse_answer(recorded["two-components"]).as_dict()
    assert [
        entry["heading"]
        for entry in parsed["ingredients"]
        if entry["kind"] == "group"
    ] == ["For the Pork Belly", "For the Kimchi Fried Rice"]


def test_the_recorded_equipment_lists_are_prose_and_not_ingredients(recorded):
    """Two of the eight answers list equipment between the ingredients and
    the instructions."""
    for name, kit in (
        ("asks-for-tips", "roasting pan"),
        ("prose-question", "paper towels"),
    ):
        parsed = parse_answer(recorded[name]).as_dict()
        names = " ".join(
            entry["text"].lower()
            for entry in parsed["ingredients"]
            if entry["kind"] == "item"
        )
        assert kit not in names
        assert kit in parsed["remainder"].lower()


def test_a_step_written_as_a_named_stage_heads_its_steps(recorded):
    """One answer writes its instructions as seven numbered stage labels over
    bullets."""
    stages = [
        step["heading"]
        for step in parse_answer(recorded["asks-for-tips"]).as_dict()["steps"]
        if step["kind"] == "group"
    ]
    assert stages[0] == "Prepare the Chicken (30 minutes)"
    assert len(stages) == 7


def test_a_three_course_answer_merges_into_one_recipe(recorded):
    """The parsed shape holds one ingredient list and one step list, so the
    one answer that writes three recipes arrives as one. Its course headings
    are kept as prose rather than dropped."""
    parsed = parse_answer(recorded["long-answer"]).as_dict()
    groups = [e for e in parsed["ingredients"] if e["kind"] == "group"]
    steps = [s for s in parsed["steps"] if s["kind"] == "step"]
    assert (len(groups), len(steps)) == (5, 22)
    assert "Course 2:" in parsed["remainder"]


def test_the_recorded_fixture_says_what_produced_it():
    """The answers are evidence about one model on one snapshot. The sidecar
    is what keeps them attributable to it."""
    sidecar = json.loads(SIDECAR.read_text())
    assert sidecar["recorded_from"].startswith("http")
    assert sidecar["endpoint"] == "POST /query"
    assert sidecar["service_info"]["models"]["generator"]
    recorded_ids = {
        json.loads(line)["id"]
        for line in RECORDINGS.read_text().splitlines()
        if line
    }
    assert set(sidecar["recordings"]) == recorded_ids
