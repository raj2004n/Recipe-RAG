"""Prompt rendering: per-query-type system + user messages for the answer drafter.

Each of the 13 query types gets its own system message. The user message is
shared across types where possible; it just lays out the chunks the model
sees and the user's query. The model returns a single JSON object
``{"answer": "<the drafted answer>"}`` on one line, parsed by the
``_AnswerRecord`` Pydantic model.

The grounded types are seeded from the ``GENERATION`` persona in
``configs/prompts_config.yaml`` but pivot away from "create a NEW recipe"
to a recommender / teacher / fact-checker role appropriate to each
retrieval shape. The six invention types keep the "create a NEW recipe"
direction; their real per-type messages are designed on a review surface,
so for now they share a placeholder. Messages are easy to edit here and
re-test in the smoke-test drafter run.
"""

import random
import time

from ollama import chat
from pydantic import BaseModel, ValidationError

from .selection import AnswerSelection
from .types import QUERY_TYPES


# Placeholder for the six invention types. The real per-type messages are
# designed on a fresh review surface; this keeps QUERY_TYPES and
# SYSTEM_MESSAGES in sync so the drafter stays importable and the coverage
# assertion passes in the meantime.
_INVENTION_PLACEHOLDER = (
    "You are a recipe developer. Using the REFERENCE RECIPES as inspiration "
    "for techniques, ingredient combinations, and proportions, create a NEW "
    "recipe that satisfies the USER QUERY. Do not reproduce an existing "
    "recipe. [PLACEHOLDER: real per-type system message designed in step 3.]"
)


# Per-type system messages. One entry per QUERY_TYPES element.
SYSTEM_MESSAGES: dict[str, str] = {
    "recipe": (
        "You are a recipe recommender for a home cook. You are given ONE "
        "retrieved recipe and a USER QUERY. Recommend that recipe in 3-6 "
        "short sentences: name the dish, say why it matches what the user "
        "asked for, and surface 1-2 distinctive details from the ingredients "
        "or instructions that make it a good fit. Do not invent a new "
        "recipe. Do not add ingredients, steps, or substitutions the user "
        "did not ask for. Ground every claim in the recipe text you were "
        "given."
    ),
    "technique": (
        "You are a cooking teacher. You are given ONE retrieved recipe and "
        "a USER QUERY about a specific technique or step in that recipe. "
        "Explain the technique in 4-8 sentences: what the step does, why it "
        "matters, and how to do it well. Quote or paraphrase the relevant "
        "instruction line. Do not invent a new recipe. Do not add techniques "
        "the user did not ask about. If the user's question is unclear, ask "
        "one short clarifying question instead of guessing."
    ),
    "info": (
        "You are a recipe fact-checker. You are given ONE info section from "
        "a recipe and a USER QUERY. Answer the user's question in 1-3 "
        "sentences using only facts from the info section. Do not invent. "
        "Do not add caveats the user did not ask about. If the info section "
        "does not actually answer the question, say so plainly in one "
        "sentence."
    ),
    "recipe_with_a_swap": (
        "You are a recipe recommender. You are given THREE recipes labelled "
        "PRIMARY, REFERENCE, and SWAP TARGET, plus a USER QUERY that asks "
        "for an ingredient swap. Recommend the primary recipe in 1-2 "
        "sentences, then walk through the swap in 3-6 sentences: which "
        "ingredient changes, what replaces it, and how the swap affects "
        "flavour or texture. Use the reference recipe to keep the swap "
        "grounded and the swap target to suggest a fitting replacement. Do "
        "not invent ingredients. Do not propose a different recipe."
    ),
    "recipe_suggestion_off_ingredients": (
        "You are a recipe recommender. You are given THREE ingredient "
        "chunks from three different recipes, plus a USER QUERY listing "
        "ingredients the user has on hand. Look at the overlap between the "
        "user's list and each recipe's ingredients, then recommend the "
        "recipe whose ingredient list best matches. In 3-6 sentences, name "
        "the recipe, list the key matching ingredients, and call out "
        "anything the user would need to add. Do not invent a new recipe. "
        "Do not recommend a recipe whose core ingredients are not in the "
        "user's list."
    ),
    "recipe_suggestion_off_instructions": (
        "You are a recipe recommender. You are given THREE instruction "
        "chunks from three different recipes, plus a USER QUERY describing "
        "a cooking method the user wants to use. Identify which recipe's "
        "method best matches the user's described setup (pan vs grill, "
        "time, equipment), and recommend it in 3-6 sentences. Explain why "
        "the method is a good fit and call out anything the user would need "
        "to adjust. Do not invent a new recipe. Do not recommend a recipe "
        "whose method actually conflicts with the user's setup."
    ),
    "recipe_suggestion_off_info": (
        "You are a recipe recommender. You are given THREE info sections "
        "from three different recipes, plus a USER QUERY describing a use "
        "case or scenario the user has in mind. Identify which recipe's "
        "notes best fit the user's described context (versatility, "
        "occasion, storage, dietary fit), then recommend it in 3-6 "
        "sentences. Explain why it fits and call out any constraint the "
        "user should know. Do not invent a new recipe. Do not recommend a "
        "recipe whose context is actually different from what the user "
        "described."
    ),
    # Invention types: placeholders until the real per-type messages are
    # designed on a review surface.
    "recipe_variation": _INVENTION_PLACEHOLDER,
    "recipe_combination": _INVENTION_PLACEHOLDER,
    "recipe_from_constraints": _INVENTION_PLACEHOLDER,
    "recipe_adaptation_method": _INVENTION_PLACEHOLDER,
    "recipe_scale": _INVENTION_PLACEHOLDER,
    "recipe_improvement": _INVENTION_PLACEHOLDER,
}


class _AnswerRecord(BaseModel):
    """The single-string JSON shape the model is asked to emit."""

    answer: str


def _user_message(selection: AnswerSelection) -> str:
    """Build the user-role message body for one AnswerSelection."""
    q = selection.query_type
    parts: list[str] = []
    if q == "recipe_with_a_swap":
        parts.append(
            "PRIMARY:\n" + selection.primary_text
            + "\n\nREFERENCE:\n" + (selection.reference_text or "")
            + "\n\nSWAP TARGET:\n" + (selection.swap_text or "")
        )
    elif q in TYPES_FOR_SIBLINGS:
        body = "PRIMARY:\n" + selection.primary_text
        for idx, sibling in enumerate(selection.sibling_texts, start=1):
            body += f"\n\nSIBLING {idx}:\n{sibling}"
        parts.append(body)
    else:
        parts.append("PRIMARY:\n" + selection.primary_text)
    parts.append("\nUSER QUERY:\n" + selection.query)
    parts.append(
        "\nRespond with a single JSON object on one line - no markdown "
        'fences, no commentary: {"answer": "<the drafted answer>"}'
    )
    return "\n".join(parts)


# Subset of QUERY_TYPES whose AnswerSelection has populated sibling_texts.
TYPES_FOR_SIBLINGS = frozenset({
    "recipe_suggestion_off_ingredients",
    "recipe_suggestion_off_instructions",
    "recipe_suggestion_off_info",
    "recipe_combination",
    "recipe_from_constraints",
})


def _strip_markdown_fence(content: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence if present.

    Some cloud models wrap their JSON in a markdown code block even when
    the request says ``format="json"``. Pydantic's strict parser rejects
    the fence, so we strip it before validating.
    """
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def call_model(
    selection: AnswerSelection,
    model_id: str,
    max_retries: int,
    backoff_base: float,
) -> str:
    """Call the model and return the parsed answer string, or "" on failure."""
    system_msg = SYSTEM_MESSAGES[selection.query_type]
    user_msg = _user_message(selection)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = chat(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
            )
            content = _strip_markdown_fence(response.message.content)
            parsed = _AnswerRecord.model_validate_json(content)
            return parsed.answer.strip()
        except (ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                sleep_for = (backoff_base ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
    if last_error is not None:
        print(f"call_model failed for type={selection.query_type}: {last_error}")
    return ""


def assert_query_types_covered() -> None:
    """Dev-time check: every entry in QUERY_TYPES has a system message."""
    missing = set(QUERY_TYPES) - set(SYSTEM_MESSAGES)
    if missing:
        raise RuntimeError(f"SYSTEM_MESSAGES missing types: {sorted(missing)}")
