"""Query-type metadata for the answer drafter.

Re-exports the prompt drafter's ``QUERY_TYPES`` tuple so the two packages
agree on the type list. Per-type answer shape lives in the prompt template
(``rendering.py``) rather than here, since the persona is what defines what
each answer must look like.
"""

from ..prompts_drafter.types import QUERY_TYPES

__all__ = ["QUERY_TYPES"]
