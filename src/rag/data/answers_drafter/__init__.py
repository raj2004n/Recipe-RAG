"""Drafts reference answers for the verified golden prompts.

Reads ``data/eval/golden_prompts.jsonl`` (the verified-prompt rows from
block 1), attaches the sibling chunks each query type needs, calls
``minimax-m3:cloud`` in parallel, and writes one row per prompt to
``data/eval/golden_answers.jsonl``. The output is resumable: a re-run
skips any (query_type, _primary_id) pair already present or recorded as
failed. The user verifies each drafted answer in the verify step.
"""

from .drafter import draft_answers
from .rendering import SYSTEM_MESSAGES
from .types import QUERY_TYPES

__all__ = ["draft_answers", "SYSTEM_MESSAGES", "QUERY_TYPES"]
