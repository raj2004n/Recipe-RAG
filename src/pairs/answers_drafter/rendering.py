import random
import time

from ollama import chat
from pydantic import BaseModel, ValidationError

from ..prompts_drafter.system_messages import answers_drafter_messages
from .selection import AnswerSelection
from .types import QUERY_TYPES


# Per-type system messages, loaded from configs/eval_system_messages.yaml.
SYSTEM_MESSAGES: dict[str, str] = answers_drafter_messages()


class _AnswerRecord(BaseModel):
    """The single-string JSON shape the model is asked to emit."""

    answer: str


# The four invention types carry the JSON line inside their system message,
# so the user message does not append it again.
_INVENTION_TYPES = frozenset({
    "recipe_variation",
    "recipe_combination",
    "recipe_from_constraints",
    "recipe_adaptation_method",
})


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
    if q not in _INVENTION_TYPES:
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
    "recipe_variation",
    "recipe_combination",
    "recipe_from_constraints",
    "recipe_adaptation_method",
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
