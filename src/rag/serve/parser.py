"""Turn a generated answer into the sections the result view draws.

The generation prompt asks for a recipe and says nothing about the headings
the model should use, so the shape of an answer is the model's choice. What
the model has actually seen is the house style of the retrieved recipes: a
bold label over dash bullets. That is what this looks for, and it is what
the recorded answers mostly are, with the deviations the model is free to
introduce: a heading or a bolded name over a sentence of preamble, grouped
ingredients under a bold sub-label, numbered steps whose lead-in is bold,
and what the recipe makes and how long it takes as their own labelled lines.

No line is dropped. A line the parser cannot place is rendered as prose
after the notes, so a parse that goes wrong costs layout and never costs
text. An answer with no ingredient, step or note section at all is not a
recipe view at all and comes back whole, because a lone heading over an
empty page reads as broken while prose reads as deliberate.

Pure over a string, so it is tested directly against recorded model output
rather than through the service.
"""

import re
from dataclasses import dataclass, replace

# The phrases that introduce a section, lowercased. The model has seen the
# house style of the retrieved recipes ("Ingredients:", "Instructions:"), so
# those are the likely shapes. A heading that starts with one of these and
# carries on ("Tips for Extra Crispness") opens the same section.
SECTION_OPENERS: dict[str, tuple[str, ...]] = {
    "ingredients": ("ingredients", "you will need", "you'll need"),
    "steps": ("instructions", "method", "steps", "directions"),
    "notes": ("notes", "tips"),
}

# The labels the model puts on what a recipe makes and how long it takes.
# Those are drawn as the badges under the title rather than left in the
# prose: they answer the two questions a cook asks before reading anything
# else, and every recorded answer carries them.
META_LABELS = frozenset(
    (
        "yields",
        "serves",
        "servings",
        "makes",
        "prep time",
        "cook time",
        "chill time",
        "chilling time",
        "total time",
        "resting time",
    )
)

# Labels the model writes over a list that is not an ingredient list. They
# are kept as prose rather than listed as ingredients, so a roasting pan
# never reads as something to cook with.
NON_INGREDIENT_LABELS = ("equipment", "tools", "utensils", "special equipment")

# A title is one line. Past these bounds the first line is prose that happens
# to come first, not the name of a dish.
TITLE_MAX_CHARS = 80
TITLE_MAX_WORDS = 12

# The label over a sub-heading of the ingredient block names one part of the
# dish ("For the Crumble Topping"). Past this bound the line is prose.
GROUP_MAX_WORDS = 8

# The labels an answer names the dish with, lowercased, alongside a markdown
# heading and a bare bold span.
NAME_LABELS = frozenset({"recipe name", "name"})

_BULLET = re.compile(r"^[-*•]\s+")
_LIST_NUMBER = re.compile(r"^\d+\s*[.)]\s*")
# A bare leading number counts as a list item, but only up to two digits and
# only with a space after it: an ingredient starting "400F" keeps its 400.
_ANY_NUMBER = re.compile(r"^\d{1,2}\s+")
_LIST_MARKER = re.compile(r"^(?:[-*•]\s+|\d+\s*[.)]\s*)")
_BOLD = re.compile(r"^\*\*(?P<inner>[^*]+?)\*\*(?P<rest>.*)$")
_BOLD_ONLY = re.compile(r"^\*\*(?P<inner>[^*]+?)\*\*:?\s*$")
_BOLD_ANY = re.compile(r"\*\*(?P<inner>[^*]+?)\*\*")
_MD_HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_MARKS = re.compile(r"\*\*")
# Emphasis the model writes with a single asterisk, which the page would
# otherwise show as punctuation.
_EMPHASIS = re.compile(r"\*([^*\n]+)\*")
_STAGE_NUMBER = re.compile(r"^\d+\s*[.)]")
_TOKEN = re.compile(r"\S+")
_NUMERIC = re.compile(r"^\d+(?:[.,]\d+)?(?:/\d+)?$|^\d+-\d+$")
# An amount written without a space, which the corpus does as often as not:
# "100g butter", "500ml milk", "2cm chunks".
_GLUED = re.compile(r"^(?P<number>\d+(?:[.,]\d+)?(?:/\d+)?)(?P<unit>[A-Za-z]+)$")

# The units the corpus writes amounts in. Enough to catch a leading
# measurement, and deliberately not a grammar: a line that starts with
# something else keeps all of its text in the name column, which is a worse
# gutter and never a wrong one.
_UNITS = frozenset(
    """
    g kg mg ml l litre litres liter liters
    tsp tsp. tbsp tbsp. teaspoon teaspoons tablespoon tablespoons
    cup cups mug mugs glass glasses
    oz ounce ounces lb lbs pound pounds
    clove cloves clove. stick sticks knob knobs head heads bunch bunches
    sprig sprigs stalk stalks slice slices rasher rashers fillet fillets
    tin tins can cans jar jars packet packets pkt sachet sachets
    pinch pinches handful handfuls dash dashes drop drops
    sheet sheets square squares cm inch inches
    """.split()
)

# Measurements that are words rather than numbers are written by the model as
# they are by the corpus, so they are matched whole instead of token by token.
_MEASUREMENT_PHRASES = (
    "to taste",
    "as needed",
    "to serve",
    "for serving",
    "for garnish",
    "for dusting",
    "a pinch",
    "a handful",
)

_FRACTIONS = "½¼¾⅓⅔⅛⅜⅝⅞"


@dataclass(frozen=True)
class MetaFact:
    """One labelled fact about the recipe as a whole, such as how many it
    serves. The label is kept exactly as the model wrote it, because it is
    what a reader recognises on the badge."""

    label: str
    value: str


@dataclass(frozen=True)
class Ingredient:
    """One ingredient, split so the amount can sit in its own column."""

    quantity: str
    text: str


@dataclass(frozen=True)
class IngredientGroup:
    """A sub-heading inside the ingredient block, such as "For the sauce".

    Kept as its own line rather than dropped: the group names what the
    ingredients below it are for, and a list that lost them would read as one
    long undifferentiated pile.
    """

    heading: str


@dataclass(frozen=True)
class StepGroup:
    """A named stage inside the step list, such as "Make the Herb Crust".

    The same idea as an ingredient group: the model writes a stage of the
    work as a label over the bullets that carry it out, and the label holds
    the timing, so it is drawn as a heading over those steps rather than
    numbered as one of them.
    """

    heading: str


@dataclass(frozen=True)
class ParsedAnswer:
    """What the result view draws. Every field may be empty."""

    title: str
    meta: tuple[MetaFact, ...]
    ingredients: tuple[Ingredient | IngredientGroup, ...]
    steps: tuple[str | StepGroup, ...]
    notes: tuple[str, ...]
    remainder: str

    @property
    def parsed(self) -> bool:
        """Whether the answer arrived in a shape worth laying out.

        A title and a row of badges on their own do not count. The fallback
        exists so that an answer the model wrote as prose reads as
        deliberate, and a lone heading over an empty page reads as broken
        instead.
        """
        return bool(self.ingredients or self.steps or self.notes)

    def as_dict(self) -> dict:
        """The parsed answer as it crosses to the page, which branches on
        `kind` to tell a group heading from an item and on `parsed` to tell a
        recipe view from the fallback to plain text."""
        ingredients = []
        for entry in self.ingredients:
            if isinstance(entry, IngredientGroup):
                ingredients.append({"kind": "group", "heading": entry.heading})
            else:
                ingredients.append(
                    {
                        "kind": "item",
                        "quantity": entry.quantity,
                        "text": entry.text,
                    }
                )

        steps = []
        for step in self.steps:
            if isinstance(step, StepGroup):
                steps.append({"kind": "group", "heading": step.heading})
            else:
                steps.append({"kind": "step", "text": step})

        return {
            "title": self.title,
            "meta": [
                {"label": fact.label, "value": fact.value} for fact in self.meta
            ],
            "ingredients": ingredients,
            "steps": steps,
            "notes": list(self.notes),
            "remainder": self.remainder,
            "parsed": self.parsed,
        }


def parse_answer(source: str) -> ParsedAnswer:
    """Parse one generated answer into the sections the result view draws."""
    lines = [line.strip() for line in source.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line]

    title, title_index = _find_title(lines)

    meta: list[MetaFact] = []
    ingredients: list[Ingredient | IngredientGroup] = []
    steps: list[str | StepGroup] = []
    notes: list[str] = []
    remainder: list[str] = []

    current: str | None = None
    for index, line in enumerate(lines):
        if index == title_index:
            continue

        section = _section_for(line)
        if section is not None:
            current = section
            continue

        fact = _meta_fact(line)
        if fact is not None:
            # The same fact stated for a second recipe in the same answer is
            # the same badge. Two different values are two facts, and both
            # are kept.
            if fact not in meta:
                meta.append(fact)
            continue

        if current in ("ingredients", "steps") and _is_group_heading(lines, index):
            # A label over the items it covers, rather than one of them.
            heading = _group_heading(line)
            if current == "ingredients":
                ingredients.append(IngredientGroup(heading=heading))
            else:
                steps.append(StepGroup(heading=heading))
        elif current is not None and _starts_an_item(line):
            if current == "ingredients":
                ingredients.append(_ingredient(line))
            elif current == "steps":
                steps.append(_plain(_strip_step_marker(line)))
            else:
                notes.append(_plain(_strip_marker(line)))
        else:
            # Text outside every recognised block. It and everything after it
            # that is not under a later opener is kept as one plain block.
            current = None
            remainder.append(_plain(line))

    answer = ParsedAnswer(
        title=title,
        meta=tuple(meta),
        ingredients=tuple(ingredients),
        steps=tuple(steps),
        notes=tuple(notes),
        remainder="\n".join(remainder),
    )

    if not answer.parsed:
        # Nothing to lay out. The answer goes back exactly as the model wrote
        # it, and the page shows it in one block with no title and no lists.
        return replace(answer, title="", meta=(), remainder=source.strip())

    if not answer.title:
        answer = replace(answer, title=_first_ingredient_line(ingredients))

    return answer


def _find_title(lines: list[str]) -> tuple[str, int | None]:
    """The name of the dish, and the line it was lifted from.

    An answer names the dish in one of three ways: a markdown heading, a
    "Recipe Name:" label, or a bold span inside its opening sentence. The
    name is looked for before the first section opener, so that a sentence of
    preamble written over the top of the recipe does not become the title,
    and a line stating what the recipe yields is never the name of it.
    """
    limit = len(lines)
    for index, line in enumerate(lines):
        if _section_for(line) is not None:
            limit = index
            break

    for index in range(limit):
        line = lines[index]
        if _meta_fact(line) is not None:
            continue

        heading = _MD_HEADING.match(line)
        if heading:
            title = _bounded(heading.group("text"))
            if title:
                return title, index

        labelled = _labelled(line)
        if labelled is not None and labelled[0].lower() in NAME_LABELS:
            title = _bounded(_plain(labelled[1]))
            if title:
                return title, index

        bold = _BOLD_ANY.search(line)
        if bold:
            title = _bounded(bold.group("inner").rstrip(":").strip())
            if title:
                return title, index

    if limit and _meta_fact(lines[0]) is None:
        title = _bare_title(lines[0])
        if title:
            return title, 0
    return "", None


def _bounded(text: str) -> str:
    """The text if it is short enough to be the name of a dish, else empty."""
    text = text.strip()
    if not text or len(text) > TITLE_MAX_CHARS:
        return ""
    return "" if len(text.split()) > TITLE_MAX_WORDS else text


def _bare_title(line: str) -> str:
    """The line as the name of a dish, if it reads as one. A sentence that
    happens to come first is prose, however short it is."""
    text = _plain(line)
    if text.endswith((".", ",", ":", ";", "?")):
        return ""
    return _bounded(text)


def _section_for(line: str) -> str | None:
    """The section this line opens, or None if it opens nothing."""
    text = _heading_text(line)
    if not text:
        return None
    lowered = text.lower()
    for section, openers in SECTION_OPENERS.items():
        for opener in openers:
            if lowered == opener or lowered.startswith(opener + " "):
                return section
    return None


def _heading_text(line: str) -> str | None:
    """The label this line is written as, or None if it is not a label.

    A label is a markdown heading, a bold span that is the whole line, or a
    short line ending in a colon. A line the model marked as a list item
    counts only when the whole of what it wrote is bold, which is how it
    heads a group of ingredients.
    """
    body = line.strip()

    heading = _MD_HEADING.match(body)
    if heading:
        return heading.group("text").strip()

    stripped = _LIST_MARKER.sub("", body, count=1)
    bold = _BOLD_ONLY.match(stripped)
    if bold:
        return bold.group("inner").strip().rstrip(":").strip()
    if stripped != body:
        return None

    if body.endswith(":") and len(body.split()) <= 5:
        return body[:-1].strip()
    return None


def _labelled(line: str) -> tuple[str, str] | None:
    """A line written as "<label>: <value>", and the two halves of it.

    The model writes these both bare and as a bullet, which is how it lists
    what a recipe makes and how long it takes.
    """
    body = _LIST_MARKER.sub("", line.strip(), count=1)

    match = _BOLD.match(body)
    if match:
        label, rest = match.group("inner"), match.group("rest")
    else:
        head, separator, rest = body.partition(":")
        if not separator or len(head.split()) > 4:
            return None
        label = head

    label = label.strip().rstrip(":").strip()
    if not label:
        return None
    return label, rest.lstrip(":").strip()


def _meta_fact(line: str) -> MetaFact | None:
    """The fact this line states about the recipe as a whole, if it states
    one. The value is dropped when the model wrote the label alone."""
    labelled = _labelled(line)
    if labelled is None or labelled[0].lower() not in META_LABELS:
        return None
    if not labelled[1]:
        return None
    return MetaFact(label=labelled[0], value=_plain(labelled[1]))


def _starts_an_item(line: str) -> bool:
    """Whether the model marked this line as a list item.

    A bare leading number counts. In an ingredient list it is the amount, and
    in a step list it is numbering the model wrote without punctuation.
    """
    return bool(
        _BULLET.match(line) or _LIST_NUMBER.match(line) or _ANY_NUMBER.match(line)
    )


def _is_group_heading(lines: list[str], index: int) -> bool:
    """Whether this line inside a block names a group of the items under it,
    rather than being one of them.

    The model marks these either by bolding the whole label or by leaving the
    bullet off. What makes it a label rather than a stray line is that the
    items it covers are listed underneath it. A numbered label may run as
    long as it likes, because numbering a stage of the work is how the model
    writes one; an unnumbered label is a name, and names are short.
    """
    text = _heading_text(lines[index])
    if not text:
        return False
    if text.lower() in NON_INGREDIENT_LABELS:
        return False
    if not _STAGE_NUMBER.match(text) and len(text.split()) > GROUP_MAX_WORDS:
        return False
    return index + 1 < len(lines) and _starts_an_item(lines[index + 1])


def _group_heading(line: str) -> str:
    """The text of a group heading, with any numbering the model put on it
    taken off, because the screen numbers the items itself."""
    return _plain(_STAGE_NUMBER.sub("", _heading_text(line) or "").strip())


def _plain(text: str) -> str:
    """The text as the page draws it, with the marks the model emphasised
    with taken off. The page renders text and never markup, so an asterisk
    that stayed would be read as one."""
    text = _MARKS.sub("", text)
    text = _EMPHASIS.sub(r"\1", text)
    return text.strip().strip("*#").strip()


def _strip_marker(line: str) -> str:
    """Remove a bullet or a numbered-list marker, keeping a bare number."""
    for pattern in (_BULLET, _LIST_NUMBER):
        match = pattern.match(line)
        if match:
            return line[match.end() :].strip()
    return line


def _strip_step_marker(line: str) -> str:
    """Remove whatever the model started a step with. Its numbering is
    discarded and re-derived, so the screen's own 1 to n is never a copy of
    whatever the model happened to type."""
    for pattern in (_BULLET, _LIST_NUMBER, _ANY_NUMBER):
        match = pattern.match(line)
        if match:
            return line[match.end() :].strip()
    return line


def _ingredient(line: str) -> Ingredient:
    quantity, text = _split_measurement(_strip_marker(line))
    return Ingredient(quantity=quantity, text=_plain(text))


def _split_measurement(line: str) -> tuple[str, str]:
    """Split an ingredient line into its amount and what it measures."""
    lowered = line.lower()
    for phrase in _MEASUREMENT_PHRASES:
        if lowered.startswith(phrase):
            return line[: len(phrase)], line[len(phrase) :].strip(" ,")

    cut = 0
    for match in _TOKEN.finditer(line):
        if not _is_measurement_token(match.group()):
            break
        cut = match.end()

    if cut == 0:
        return "", line
    return line[:cut].strip(), line[cut:].strip()


def _is_measurement_token(token: str) -> bool:
    bare = token.strip(".,;()")
    if not bare:
        return False
    if bare.lower() in _UNITS:
        return True
    if bare in ("x", "×", "-", "–"):
        return True
    if bare[0] in _FRACTIONS:
        return True
    glued = _GLUED.match(bare)
    if glued:
        return glued.group("unit").lower() in _UNITS
    return bool(_NUMERIC.match(bare))


def _first_ingredient_line(
    ingredients: list[Ingredient | IngredientGroup],
) -> str:
    """The first line of the ingredient block, used as the title when the
    answer opened with prose instead of a name."""
    for entry in ingredients:
        if isinstance(entry, IngredientGroup):
            return entry.heading
        return " ".join(part for part in (entry.quantity, entry.text) if part)
    return ""
