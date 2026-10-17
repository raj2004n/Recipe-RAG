import hashlib
import logging
import random
import re
import time

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from ..domain.chunker import ingredient_chunks, instruction_chunks
from ..domain.types import Recipe
from .types import (
    BACKOFF_BASE,
    MAX_RETRIES,
    META_PHRASES,
    MODEL_ID,
    NUM_CTX,
    NUM_PREDICT,
    QUANTITY_RE,
    SEED,
    STEP_RE,
    TEMPERATURE,
    THINK,
    TITLE_STOPWORDS,
    WORDS_MAX,
    WORDS_MIN,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You design one writing prompt from a pair of recipes.

You are given two recipes, each with its name, its ingredients and its instructions. Read both
before you write anything, and find what they share. What they share is the raw material for the
prompt you are about to write.

Your prompt is not for you. It goes to another model, which will read it alongside the same two
recipes, and that model has to invent one new recipe from it. Your prompt is the only steering
it gets, so everything that matters has to be in it.

Write a prompt that asks for a recipe neither of these two is. Not a variation, not a lighter or
faster version, not the two of them stacked together. The shared thing is the starting point,
not the destination: keep what makes it work, then move somewhere this pair does not go.

Pick the one move the pair actually supports:

- change the format, so the same flavours land as a different thing on the plate
- change the cooking method, and with it the texture
- move the cuisine or the flavour register along its own logic
- put the shared base to a different job in the meal
- keep the shared base and change what carries it
- add a constraint that forces invention: one pan, no oven, twenty minutes, a weeknight, a
  lunchbox, nothing that needs a special trip to the shop
- take the contrast that made these two interesting side by side, and make one dish out of it
  instead of two

Choose one axis, the one this pair supports. A prompt that could have been written from any
other pair is a wasted prompt. If the two recipes share only a cuisine or a single ingredient,
work from that, and if all they share is that both are food, work from the stronger of the two.

The prompt you write must:

- read like a person asking a cooking assistant, one to three sentences, ordinary language
- name the thing to invent, so the other model cannot answer by rewriting one of the two recipes
- give enough grip to be answerable: the course or the format, two to four components or
  techniques, the flavour direction, and one constraint if the pair suggests one
- stay out of the kitchen's way: no quantities, no ingredient list, no numbered steps, no method
- be one request for one recipe, not a menu, not options, not a list of ideas
- stand on its own, with nothing about how to answer it and nothing about a task, a model or
  a set of rules

You may point at the two recipes as "these two", because the other model sees them too. What you
may never do is name either dish, quote either title, name an amount, describe a step, or ask for
a mashup of two named dishes. Ask for a dish that could only have come from understanding both.

Never write the recipe yourself. You are writing the request, not the answer.

Reply with JSON only:
{"prompt": "<the prompt, one to three sentences>", "axis": "<the one move you chose, in a few words>"}

Everything inside angle brackets is a placeholder to be replaced. Return the prompt and the move,
nothing else."""


class Design(BaseModel):
    """The designer's JSON response for one pair."""

    prompt: str = Field(min_length=1, max_length=1200)
    axis: str = Field(default="", max_length=200)


def strip_markdown_fence(content: str) -> str:
    """Drop a leading and trailing ``` fence, which some models add anyway."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def prompt_hash() -> str:
    """A stable fingerprint of the instructions the run sends, for the manifest."""
    return "sha256:" + hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def ingredients_text(recipe: Recipe) -> str:
    """A recipe's ingredients, rendered as the sparse index renders them."""
    return ingredient_chunks([recipe])[0].text


def instructions_text(recipe: Recipe) -> str:
    """A recipe's instructions, rendered as the sparse index renders them."""
    return instruction_chunks([recipe])[0].text


def recipe_block(label: str, recipe: Recipe) -> str:
    """One recipe as the designer reads it: the name it goes by, then its two working sections.

    Servings, notes and info sections are left out. They say nothing about what the dish is, and a
    designer reading them would be reading the page rather than the cooking.
    """
    return (
        f"{label}: {recipe.recipe_name}\n\n"
        f"{ingredients_text(recipe)}\n\n{instructions_text(recipe)}"
    )


def user_message(primary: Recipe, partner: Recipe) -> str:
    """The call's user turn: both recipes, and nothing else."""
    return f"{recipe_block('RECIPE 1', primary)}\n\n{recipe_block('RECIPE 2', partner)}"


def title_phrases(name: str) -> set[str]:
    """The two-word phrases in a dish name that identify it, stopwords stripped from the ends.

    Phrases rather than single words, because a prompt that asks for chicken is doing its job
    while a prompt that asks for "lemonade scones" has copied the dish it was designing from.
    """
    tokens = [token for token in re.findall(r"[a-z0-9]+", name.lower())]
    return {
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if left not in TITLE_STOPWORDS and right not in TITLE_STOPWORDS
    }


def flag_prompt(prompt: str, primary_name: str, partner_name: str) -> list[str]:
    """What a read-through of this prompt would raise, in the order a reviewer meets it."""
    flags = []
    words = len(prompt.split())
    if words < WORDS_MIN or words > WORDS_MAX:
        flags.append(f"length {words} words, outside the {WORDS_MIN} to {WORDS_MAX} band")

    lowered = prompt.lower()
    for name in (primary_name, partner_name):
        copied = sorted(phrase for phrase in title_phrases(name) if phrase in lowered)
        if copied:
            flags.append(f"names a source dish: {', '.join(copied)}")
    for phrase in META_PHRASES:
        if phrase in lowered:
            flags.append(f"reads as task language: {phrase}")
    if QUANTITY_RE.search(prompt):
        flags.append("quotes a quantity")
    if STEP_RE.search(prompt):
        flags.append("numbers steps")
    return flags


class DesignerClient:
    """The chat call, its retries, and the tokens it spends.

    The retry covers what the provider produces in practice: a response wrapped in a markdown
    fence, and a response that does not validate against the shape.
    """

    def __init__(self, model_id: str = MODEL_ID, temperature: float = TEMPERATURE,
                 seed: int | None = SEED, think: bool = THINK, num_ctx: int = NUM_CTX,
                 num_predict: int = NUM_PREDICT, max_retries: int = MAX_RETRIES,
                 backoff_base: float = BACKOFF_BASE):
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _options(self) -> dict:
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def raw_call(self, message: str):
        """One call, returned whole, for a probe that inspects the wire shape."""
        return chat(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            think=self.think,
            format="json",
            options=self._options(),
        )

    def design(self, message: str) -> tuple[Design | None, dict]:
        """The validated design, with the tokens and the latency it took."""
        usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0, "retries": 0}
        started = time.monotonic()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.raw_call(message)
                usage["calls"] += 1
                usage["prompt_tokens"] += response.prompt_eval_count or 0
                usage["output_tokens"] += response.eval_count or 0
                design = Design.model_validate_json(
                    strip_markdown_fence(response.message.content or "")
                )
                usage["seconds"] = round(time.monotonic() - started, 2)
                return design, usage
            except (ValidationError, ValueError, Exception) as error:  # noqa: BLE001
                last_error = error
                usage["retries"] += 1
                if attempt < self.max_retries - 1:
                    # Jitter so parallel workers do not retry in lockstep.
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("design call failed for one pair: %s", last_error)
        usage["error"] = str(last_error)
        usage["seconds"] = round(time.monotonic() - started, 2)
        return None, usage
