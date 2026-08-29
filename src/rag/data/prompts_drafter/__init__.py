"""Drafts the user-question prompts half of the eval-data golden subset.

For each of thirteen query types, the drafter selects context chunks from the
recipe corpus using the build encoder for similarity, then asks a model to
produce a single user-style query string per selection. Output rows are
written to ``data/eval/prompts.jsonl`` and carry the full context they were
generated from, so the answer-drafting step can re-use the same context
without re-fetching.

The drafter is resumable: rows already present for a (query_type, primary_id)
pair are skipped on re-run. The new file is written to a sibling temp path
and atomically renamed over the existing output, so an interrupted run
leaves the previous output in place rather than moving it aside.
"""

from .drafter import draft_prompts
from .types import QUERY_TYPES, TYPES_WITH_ONE_PRIMARY

__all__ = ["draft_prompts", "QUERY_TYPES", "TYPES_WITH_ONE_PRIMARY"]
