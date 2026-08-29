"""Prompt rendering: shape each Selection into the message the model sees.

This module owns the per-query-type prompt templates and the call to the
model. It reads from ``Selection`` objects but does not build them - that
is selection.py's job.
"""

import random
import time

from ollama import chat
from pydantic import BaseModel, ValidationError

from .selection import Selection
from .system_messages import prompts_drafter_messages
from .types import TYPES_WITH_ONE_PRIMARY


# Per-type system messages, loaded from configs/eval_system_messages.yaml. The
# suggestion types' {label} placeholder is formatted at load from
# QUERY_TYPE_SPECS so the two stay in sync.
SYSTEM_MESSAGES: dict[str, str] = prompts_drafter_messages()


class _PromptRecord(BaseModel):
    """The single-string JSON shape the model is asked to emit."""

    query: str


def user_message(selection: Selection) -> str:
    """Build the user-role message body for a single selection."""
    if selection.query_type in TYPES_WITH_ONE_PRIMARY:
        return _single_block_user_message(selection)
    return _multi_block_user_message(selection)


def _single_block_user_message(selection: Selection) -> str:
    label_map = {
        "recipe": "RECIPE",
        "technique": "RECIPE",
        "info": "INFO SECTION",
    }
    return f"{label_map[selection.query_type]}:\n{selection.primary_text}"


def _multi_block_user_message(selection: Selection) -> str:
    if selection.query_type == "recipe_with_a_swap":
        return (
            f"PRIMARY:\n{selection.primary_text}\n\n"
            f"REFERENCE:\n{selection.reference_text or ''}\n\n"
            f"SWAP TARGET:\n{selection.swap_text or ''}"
        )
    if selection.query_type in {"recipe_variation", "recipe_adaptation_method"}:
        # The primary is the recipe the query adapts; the single sibling is the
        # reference the answer should draw from.
        return (
            f"PRIMARY:\n{selection.primary_text}\n\n"
            f"SIBLING:\n{selection.sibling_texts[0]}"
        )
    if selection.query_type in {"recipe_combination", "recipe_from_constraints"}:
        return (
            f"RECIPE 1:\n{selection.primary_text}\n\n"
            f"RECIPE 2:\n{selection.sibling_texts[0]}"
        )
    # The three recipe_suggestion_off_* types.
    return (
        f"CHUNK 1:\n{selection.primary_text}\n\n"
        f"CHUNK 2:\n{selection.sibling_texts[0]}\n\n"
        f"CHUNK 3:\n{selection.sibling_texts[1]}"
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
    selection: Selection,
    model_id: str,
    max_retries: int,
    backoff_base: float,
) -> str:
    """Call the model and return the parsed query string, or "" on failure."""
    user_msg = user_message(selection)
    system_msg = SYSTEM_MESSAGES[selection.query_type]
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
        print(f"call_model failed for type={selection.query_type}: {last_error}")
    return ""
