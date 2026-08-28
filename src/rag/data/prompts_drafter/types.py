"""Query-type metadata for the eval-prompt drafter.

The drafter has nine query types and a small number of properties that vary
per type. This module is the single source of truth for that metadata; the
selection, rendering, and orchestration submodules all read from it.
"""

from dataclasses import dataclass


QUERY_TYPES: tuple[str, ...] = (
    "recipe",
    "technique",
    "recipe_with_a_swap",
    "info",
    "technique_suggestion",
    "recipe_suggestion_off_ingredients",
    "recipe_suggestion_off_instructions",
    "recipe_suggestion_off_info",
    "ingredient_suggestion",
)


@dataclass(frozen=True)
class QueryTypeSpec:
    """One row of the query-type metadata table.

    Keeping these properties on a single source of truth stops the
    selection branches, the context-chunks writer, and the per-type prompt
    templates from drifting.
    """

    name: str
    requires_siblings: bool
    # Used by recipe_suggestion_off_*; ignored otherwise. Rendered as the
    # {label} placeholder in their multi-block prompt body.
    label: str | None = None


QUERY_TYPE_SPECS: tuple[QueryTypeSpec, ...] = (
    QueryTypeSpec("recipe", requires_siblings=False),
    QueryTypeSpec("technique", requires_siblings=False),
    QueryTypeSpec("recipe_with_a_swap", requires_siblings=True),
    QueryTypeSpec("info", requires_siblings=False),
    QueryTypeSpec("technique_suggestion", requires_siblings=False),
    QueryTypeSpec("recipe_suggestion_off_ingredients", requires_siblings=True, label="shared ingredients"),
    QueryTypeSpec("recipe_suggestion_off_instructions", requires_siblings=True, label="shared cooking method"),
    QueryTypeSpec("recipe_suggestion_off_info", requires_siblings=True, label="shared notes or meta info"),
    QueryTypeSpec("ingredient_suggestion", requires_siblings=True),
)

TYPES_WITH_ONE_PRIMARY: frozenset[str] = frozenset(
    s.name for s in QUERY_TYPE_SPECS if not s.requires_siblings
)

# A negative chunk_id reserved for selections that don't correspond to a chunker
# chunk (the per-recipe info section, which the chunker emits per-entry rather
# than per-section). Reserving a sentinel range keeps primary_id lookups stable.
INFO_SECTION_CHUNK_ID_BASE = -1
