"""Prompt rendering: shape each Selection into the message the model sees.

This module owns the per-query-type prompt templates and the call to the
model. It reads from ``Selection`` objects but does not build them - that
is selection.py's job.
"""

import random
import time

from ollama import chat
from pydantic import BaseModel, ValidationError

from .types import TYPES_WITH_ONE_PRIMARY


SYSTEM_PROMPT = (
    "You are a home cook drafting questions to test a recipe retrieval system. "
    "Produce exactly one user-style query string. Your entire response must be "
    "a single JSON object on one line - no markdown fences, no commentary, "
    'no surrounding text. Exact shape: {"query": "<the query>"}'
)


# Per-type system messages for the invention types. The grounded types keep
# the shared SYSTEM_PROMPT until their per-type messages are designed. Each
# message carries the JSON line, so the user message does not append it again.
INVENTION_SYSTEM_MESSAGES: dict[str, str] = {
    "recipe_variation": (
        "You are a home cook drafting questions to test a recipe retrieval "
        "system. You are given ONE primary recipe and ONE sibling recipe close "
        "to it. Write a single user-style query that names the primary recipe "
        "and asks how to adapt it to a constraint: a dietary need, an excluded "
        "ingredient, or a flavor change. The query should read like a real home "
        "cook asking for help, reference what makes the recipe distinctive, and "
        "be answerable only with the primary recipe plus the sibling as a "
        "reference. Do not quote the recipe name verbatim.\n\n"
        "Produce exactly one user-style query string. Your entire response must "
        "be a single JSON object on one line - no markdown fences, no "
        'commentary, no surrounding text. Exact shape: {"query": "<the query>"}'
    ),
    "recipe_combination": (
        "You are a home cook drafting questions to test a recipe retrieval "
        "system. You are given TWO recipes. Write a single user-style query "
        "that names both dishes and asks to fuse them into one new dish, "
        "drawing techniques, ingredients, and proportions from both. The query "
        "should read like a real home cook asking for help and be answerable "
        "only with both recipes in context. Do not quote either recipe name "
        "verbatim.\n\n"
        "Produce exactly one user-style query string. Your entire response must "
        "be a single JSON object on one line - no markdown fences, no "
        'commentary, no surrounding text. Exact shape: {"query": "<the query>"}'
    ),
    "recipe_from_constraints": (
        "You are a home cook drafting questions to test a recipe retrieval "
        "system. You are given TWO recipes. Write a single user-style query "
        "that gives a constraint set (ingredients on hand, equipment, time, "
        "dietary) that no single recipe satisfies, and asks for a new recipe "
        "assembled from these sources. The query should read like a real home "
        "cook asking for help and be answerable only with both recipes in "
        "context. Do not quote either recipe name verbatim.\n\n"
        "Produce exactly one user-style query string. Your entire response must "
        "be a single JSON object on one line - no markdown fences, no "
        'commentary, no surrounding text. Exact shape: {"query": "<the query>"}'
    ),
    "recipe_adaptation_method": (
        "You are a home cook drafting questions to test a recipe retrieval "
        "system. You are given ONE primary recipe and ONE sibling recipe that "
        "uses a different cooking method. Write a single user-style query that "
        "names the primary recipe and asks how to convert it to that method "
        "(oven to air fryer, stovetop to slow cooker, and so on). The query "
        "should read like a real home cook asking for help, reference what "
        "makes the recipe distinctive, and be answerable only with the primary "
        "recipe plus the sibling as a reference. Do not quote the recipe name "
        "verbatim.\n\n"
        "Produce exactly one user-style query string. Your entire response must "
        "be a single JSON object on one line - no markdown fences, no "
        'commentary, no surrounding text. Exact shape: {"query": "<the query>"}'
    ),
}


class _PromptRecord(BaseModel):
    """The single-string JSON shape the model is asked to emit."""

    query: str


def user_message(query_type: str, primary_text: str, sibling_texts: list[str]) -> str:
    """Build the user-role message body for a single (type, primary) selection."""
    if query_type in TYPES_WITH_ONE_PRIMARY:
        body = _single_block_user_message(query_type, primary_text)
    else:
        body = _multi_block_user_message(query_type, primary_text, sibling_texts)
    if query_type in INVENTION_SYSTEM_MESSAGES:
        # The invention system messages carry the JSON line; the user message
        # is just the chunks.
        return body
    return (
        body
        + "\n\nRespond with a single JSON object on one line - no markdown fences, "
        + 'no commentary: {"query": "<the query>"}'
    )


def _single_block_user_message(query_type: str, primary_text: str) -> str:
    label_map = {
        "recipe": "RECIPE",
        "technique": "RECIPE",
        "info": "INFO SECTION",
    }
    label = label_map[query_type]
    instruction_map = {
        "recipe": (
            "Generate a short question a home cook might type to find this exact recipe. "
            "Reference what makes it distinctive (an ingredient, technique, cuisine, or occasion)."
        ),
        "technique": (
            "Read the instructions. Pick one notable technique they describe and ask how to do it. "
            "Don't quote the recipe name verbatim."
        ),
        "info": (
            "Generate a meta question about the section above - spiciness, cuisine, difficulty, "
            "timing, or serving. Don't reference the recipe by name."
        ),
    }
    return f"{label}:\n{primary_text}\n\n{instruction_map[query_type]}"


def _multi_block_user_message(query_type: str, primary_text: str, sibling_texts: list[str]) -> str:
    if query_type == "recipe_with_a_swap":
        return (
            "Below are three similar recipes. The first is the primary; the second is a reference; "
            "the third is the swap target. Write a prompt that asks the home cook how to substitute "
            "one ingredient from the primary with something from the swap target, using the reference "
            "to keep the swap grounded.\n\n"
            f"PRIMARY:\n{primary_text}\n\n"
            f"REFERENCE:\n{sibling_texts[0]}\n\n"
            f"SWAP TARGET:\n{sibling_texts[1]}"
        )
    if query_type in {"recipe_variation", "recipe_adaptation_method"}:
        # The primary is the recipe the query adapts; the single sibling is the
        # reference the answer should draw from.
        return (
            f"PRIMARY:\n{primary_text}\n\n"
            f"SIBLING:\n{sibling_texts[0]}"
        )
    if query_type in {"recipe_combination", "recipe_from_constraints"}:
        return (
            f"RECIPE 1:\n{primary_text}\n\n"
            f"RECIPE 2:\n{sibling_texts[0]}"
        )
    # The three recipe_suggestion_off_* types.
    label_map = {
        "recipe_suggestion_off_ingredients": "shared ingredients",
        "recipe_suggestion_off_instructions": "shared cooking method",
        "recipe_suggestion_off_info": "shared notes or meta info",
    }
    return (
        f"Below are three chunks from three different recipes that share {label_map[query_type]}. "
        "Write a 'what can I make' prompt grounded in the overlap - don't name a specific recipe.\n\n"
        f"CHUNK 1:\n{primary_text}\n\n"
        f"CHUNK 2:\n{sibling_texts[0]}\n\n"
        f"CHUNK 3:\n{sibling_texts[1]}"
    )


def prompt_template_tag(query_type: str) -> str:
    return f"{query_type}_v1"


def _strip_markdown_fence(content: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence if present.

    Some cloud models wrap their JSON in a markdown code block even when the
    request says ``format="json"``. Pydantic's strict parser rejects the fence,
    so we strip it before validating.
    """
    s = content.strip()
    if s.startswith("```"):
        # Drop the opening fence line (```json or ```), then any closing fence.
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def call_model(
    query_type: str,
    primary_text: str,
    sibling_texts: list[str],
    model_id: str,
    max_retries: int,
    backoff_base: float,
) -> str:
    """Call the model and return the parsed query string, or "" on failure."""
    user_msg = user_message(query_type, primary_text, sibling_texts)
    system_msg = INVENTION_SYSTEM_MESSAGES.get(query_type, SYSTEM_PROMPT)
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
            parsed = _PromptRecord.model_validate_json(content)
            return parsed.query.strip()
        except (ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                # Jitter so parallel workers don't retry in lockstep.
                sleep_for = (backoff_base ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
    if last_error is not None:
        print(f"call_model failed for type={query_type}: {last_error}")
    return ""
