import logging
import random
import re
import time
from collections.abc import Sequence

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from ..data.recipe_format import recipe_to_string
from .judge_prompts import judge_messages
from .scoring import aggregate, budget_texts, judgment


logger = logging.getLogger(__name__)


# Structural thresholds. The golden reference answers run roughly 1,400 to
# 3,300 characters, carry well over a dozen quantity-bearing ingredient lines,
# and number their steps; these bounds are loose enough that any answer in the
# reference family passes and only real violations fail.
MIN_INGREDIENT_LINES = 3
MIN_STEPS = 3
MIN_RESPONSE_CHARS = 300
MAX_RESPONSE_CHARS = 8000

# Novelty thresholds. The persona demands a NEW recipe, so a near-verbatim
# echo of a reference is a failure even though every other check might pass.
# The shingle overlap stays high (long shared word runs) because an adapted
# recipe legitimately reuses ingredient lines and even whole steps from its
# references.
_NOVELTY_SHINGLE = 10
_NOVELTY_MAX_OVERLAP = 0.9

_QUANTITY = re.compile(
    r"\d|¼|½|¾|⅓|⅔|clove|pinch|splash|handful|sprig|to taste"
)
_NUMBERED_STEP = re.compile(r"^\s*\d+\s*[.)]")
_MARKDOWN_FENCE = "```"
_MAX_NAME_CHARS = 100


class CoreVerdict(BaseModel):
    """The core judge persona's JSON response."""

    constraint_satisfaction: int = Field(ge=0, le=2)
    query_relevance: int = Field(ge=0, le=2)
    faithfulness: int = Field(ge=0, le=2)
    reasons: dict[str, str] = Field(default_factory=dict)


class ExpectednessVerdict(BaseModel):
    """The expectedness judge persona's JSON response."""

    expectedness: int = Field(ge=0, le=2)
    reason: str = ""


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _shingle_overlap(response_words: list[str], reference_words: list[str], n: int) -> float:
    """Fraction of the response's word n-grams that also occur in the reference."""
    if len(response_words) < n or len(reference_words) < n:
        return 0.0
    reference = {
        tuple(reference_words[i:i + n])
        for i in range(len(reference_words) - n + 1)
    }
    total = len(response_words) - n + 1
    hits = sum(
        1
        for i in range(total)
        if tuple(response_words[i:i + n]) in reference
    )
    return hits / total


def _section_header(line: str) -> str | None:
    """The section a header line names, with markdown decoration tolerated.

    Generated answers mark sections variously: plain "Ingredients", markdown
    "## Ingredients", or bold "**Ingredients:**". The decoration is stripped
    before matching so a decorated header still opens its section.
    """
    cleaned = (
        line.strip().lstrip("#").strip().strip("*").strip().rstrip(":").strip().lower()
    )
    for name in ("ingredients", "instructions", "method", "steps"):
        if cleaned == name or cleaned.startswith(name + " "):
            return name
    return None


def _ingredient_lines(response: str) -> list[str]:
    """The non-blank lines of the ingredients section, headers excluded.

    The section runs from the first ingredients heading to the first
    instructions heading (or the end of the text). Subsection names inside
    the section ("Meatballs", "Sauce") are plain lines and are counted here;
    the quantity check below tolerates a handful of them, matching the
    reference answers' own layout.
    """
    lines = response.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _section_header(line) == "ingredients"),
        None,
    )
    if start is None:
        return []
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if _section_header(lines[i]) in ("instructions", "method", "steps")
        ),
        len(lines),
    )
    return [
        line
        for line in (raw.strip() for raw in lines[start + 1:end])
        if line and not line.startswith("#")
    ]


def structural_checks(response: str, reference_texts: Sequence[str]) -> dict:
    """Deterministic format and novelty checks on one generated answer.

    Returns a flag dict. Every flag is True when the answer satisfies that
    part of the persona's format contract; structure_pass is True only when
    all flags hold. These checks run before any judge call so a formatting
    failure is visible in the per-row record even when the judged scores look
    fine, and the judge is told to ignore formatting so the two never double
    count the same defect.
    """
    flags = {}
    lines = [line for line in (raw.strip() for raw in response.splitlines()) if line]
    name_line = lines[0] if lines else ""

    # The persona contract wants the recipe to open on its name, so a chatty
    # lead-in sentence fails the check. A name is short and does not end on
    # sentence punctuation; decorated name lines ("**Recipe Name:** ...")
    # are fine.
    flags["has_name"] = bool(
        name_line
        and len(name_line) <= _MAX_NAME_CHARS
        and not name_line.rstrip("*").strip().endswith((".", "!", "?"))
        and _MARKDOWN_FENCE not in name_line
        and _section_header(name_line) != "ingredients"
    )

    ingredient_lines = _ingredient_lines(response)
    quantity_lines = [
        line for line in ingredient_lines if _QUANTITY.search(line)
    ]
    flags["ingredients_with_amounts"] = (
        len(quantity_lines) >= MIN_INGREDIENT_LINES
        and len(quantity_lines) * 2 >= len(ingredient_lines)
    )

    flags["numbered_steps"] = (
        sum(1 for line in lines if _NUMBERED_STEP.match(line)) >= MIN_STEPS
    )

    flags["no_markdown_fence"] = _MARKDOWN_FENCE not in response

    flags["length_in_bounds"] = MIN_RESPONSE_CHARS <= len(response) <= MAX_RESPONSE_CHARS

    flags["not_verbatim_copy"] = not _is_verbatim_copy(response, reference_texts)

    flags["structure_pass"] = all(
        value for key, value in flags.items() if key != "structure_pass"
    )
    return flags


def _is_verbatim_copy(response: str, reference_texts: Sequence[str]) -> bool:
    """Whether the response echoes any reference recipe near-verbatim.

    A copy is either literal containment after whitespace normalization, or a
    near-total overlap of long word runs. Shorter overlaps are legitimate:
    adapted recipes reuse ingredient lines and steps from their references.
    """
    response_norm = _normalize(response)
    response_words = response_norm.split()
    for reference in reference_texts:
        reference_norm = _normalize(reference)
        if not reference_norm:
            continue
        if response_norm in reference_norm or reference_norm in response_norm:
            return True
        overlap = _shingle_overlap(
            response_words, reference_norm.split(), _NOVELTY_SHINGLE
        )
        if overlap > _NOVELTY_MAX_OVERLAP:
            return True
    return False


def _strip_markdown_fence(content: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence if present.

    Some cloud models wrap their JSON in a markdown code block even when the
    request says format="json". Pydantic's strict parser rejects the fence,
    so strip it before validating.
    """
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


class JudgeClient:
    """Ollama-chat client for the two judge personas.

    Rides the same chat-with-format-json pattern as the eval drafters, adding
    the options dict for sampling control: a small nonzero temperature so two
    calls can genuinely disagree, an explicit num_ctx sized for the packed
    judge prompt (query plus two whole recipes plus the generated answer, and
    for expectedness the reference answer too), a num_predict cap, and an
    optional seed. Judge calls that fail to parse or validate are retried with
    backoff; a row whose calls all fail is recorded with no score rather than
    dropping the row, so the per-row file stays uniform.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        num_ctx: int = 16384,
        num_predict: int = 800,
        seed: int | None = None,
        max_retries: int = 3,
        backoff_base: float = 2,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.messages = judge_messages()

    def _call(self, persona: str, user_message: str, model: type[BaseModel]):
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = chat(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.messages[persona]},
                        {"role": "user", "content": user_message},
                    ],
                    format="json",
                    options=options,
                )
                content = _strip_markdown_fence(response.message.content)
                return model.model_validate_json(content)
            except (ValidationError, Exception) as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base ** attempt + random.uniform(0, 0.5))
        logger.warning("judge call failed: persona=%s error=%s", persona, last_error)
        return None

    def score_core(self, query: str, context: str, response: str):
        """Score the three content dimensions; no reference answer shown."""
        return self._call(
            "core", _core_user_message(query, context, response), CoreVerdict
        )

    def score_expectedness(self, query: str, reference_answer: str, response: str):
        """Score whether the response captures the point of the reference."""
        return self._call(
            "expectedness",
            _expectedness_user_message(query, reference_answer, response),
            ExpectednessVerdict,
        )


def _core_user_message(query: str, context: str, response: str) -> str:
    return (
        "User request:\n" + query
        + "\n\nReference recipes:\n" + context
        + "\n\nGenerated recipe:\n" + response
        + "\n\nScore the generated recipe. JSON only."
    )


def _expectedness_user_message(
    query: str, reference_answer: str, response: str
) -> str:
    return (
        "User request:\n" + query
        + "\n\nReference answer (one valid instantiation):\n" + reference_answer
        + "\n\nGenerated recipe:\n" + response
        + "\n\nScore whether the generated recipe captures the point of the "
          "reference answer. JSON only."
    )


class GeneratorEvaluator:
    """Scores the generator on golden rows whose context is given, not retrieved.

    For every row the ground-truth recipe_ids are resolved against the corpus
    (recipes_by_id), the serving prompt is built with the injected
    prompt_builder and generator, and the answer is scored with structural
    checks plus the double-called judge. The prompt_builder and generator are
    injected so the evaluator tests whatever serving configuration it is
    given, and the judge is any object with the JudgeClient's two score
    methods, which keeps the evaluator testable with stub judges.
    """

    def __init__(
        self,
        prompt_builder,
        generator,
        judge,
        recipes_by_id: dict[int, dict],
        judge_calls: int = 2,
    ):
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.judge = judge
        self.recipes_by_id = recipes_by_id
        self.judge_calls = judge_calls

    def evaluate(self, rows: Sequence[dict]) -> dict:
        """Score every row and aggregate over the slice and per query type.

        Rows must all carry a verified reference answer, since the
        expectedness judge needs one. The slice aggregate holds the mean of
        every judged dimension (a row's score being the mean of its judge
        calls), the per-dimension disagreement rate, and the structural flag
        rates; the query_type aggregates are the same over each type's rows.
        """
        per_row = [self._evaluate_row(row) for row in rows]
        query_types = sorted({row.get("query_type") for row in per_row})
        return {
            "per_row": per_row,
            "slice": self._aggregate(per_row),
            "query_type": {
                query_type: self._aggregate(
                    [r for r in per_row if r["query_type"] == query_type]
                )
                for query_type in query_types
            },
        }

    def _evaluate_row(self, row: dict) -> dict:
        query = row["query"]
        recipe_ids = list(row["recipe_ids"])
        reference_answer = row.get("answer")

        recipes = [self.recipes_by_id.get(rid) for rid in recipe_ids]
        missing = [
            rid for rid, recipe in zip(recipe_ids, recipes) if recipe is None
        ]
        if missing:
            logger.warning("row has recipe_ids missing from the corpus: %s", missing)

        resolved = [recipe for recipe in recipes if recipe is not None]
        texts = [recipe_to_string(recipe) for recipe in resolved]
        included = self._budgeted_texts(texts)
        context = "\n\n".join(included)

        system, user = self.prompt_builder.build_prompt_from_recipes(
            query=query,
            recipes=resolved,
        )
        response = self.generator.generate(system, user)

        flags = structural_checks(response, texts)

        core_verdicts = [
            self.judge.score_core(query, context, response)
            for _ in range(self.judge_calls)
        ]
        expectedness_verdicts = (
            [
                self.judge.score_expectedness(query, reference_answer, response)
                for _ in range(self.judge_calls)
            ]
            if reference_answer
            else []
        )

        scores, disagreement = judgment(core_verdicts, expectedness_verdicts)

        return {
            "query": query,
            "query_type": row.get("query_type"),
            "recipe_ids": recipe_ids,
            "missing_recipe_ids": missing,
            "context_truncated": len(included) < len(texts),
            "generated": response,
            "structural": flags,
            "scores": scores,
            "disagreement": disagreement,
            "core_judge": [
                v.model_dump() if v else None for v in core_verdicts
            ],
            "expectedness_judge": [
                v.model_dump() if v else None for v in expectedness_verdicts
            ],
        }

    def _budgeted_texts(self, texts: Sequence[str]) -> list[str]:
        """The texts the prompt builder will keep under its token budget."""
        return budget_texts(
            texts,
            self.prompt_builder.tokenizer,
            self.prompt_builder.max_context_tokens,
        )

    @classmethod
    def _aggregate(cls, rows: Sequence[dict]) -> dict:
        """Slice-level means, disagreement rates, and structural flag rates."""
        return aggregate(rows)