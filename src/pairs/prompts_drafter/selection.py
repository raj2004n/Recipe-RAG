import random
from dataclasses import dataclass, field
from typing import Any

from ...domain.chunker import (
    Chunk,
    instruction_chunks,
    ingredient_chunks,
    whole_recipe_chunks,
)
from ...build.encoder import Encoder
from ...corpus.preprocessor import Recipe
from .types import INFO_SECTION_CHUNK_ID_BASE


@dataclass
class Selection:
    """One (type, primary) selection, with the full context the model will see.

    For multi-block query types, ``sibling_texts`` carries the two sibling
    chunks in their prompt order. ``recipe_with_a_swap`` is the exception: the
    siblings have distinct semantic roles (reference vs. swap target), so it
    populates ``reference_text`` / ``swap_text`` and leaves ``sibling_texts``
    empty. Downstream code checks ``query_type`` directly to decide which
    fields are meaningful.
    """

    query_type: str
    primary_id: int
    primary_text: str
    primary_recipe_id: int
    sibling_texts: list[str] = field(default_factory=list)
    sibling_recipe_ids: list[int] = field(default_factory=list)
    # Used only by recipe_with_a_swap.
    reference_text: str | None = None
    swap_text: str | None = None


def _build_whole_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return whole_recipe_chunks(recipes)


def _build_ingredient_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return ingredient_chunks(recipes)


def _build_instruction_chunks(recipes: list[Recipe]) -> list[Chunk]:
    return instruction_chunks(recipes)


@dataclass
class _InfoSection:
    """Per-recipe wrapper around the chunker's per-entry info chunks.

    The chunker emits one chunk per info entry; the ``info`` query type
    selects one chunk per recipe's complete info section. ``_InfoSection``
    joins the recipe's entries into a single block with a stable negative
    ``primary_id`` so resume-key lookups stay unique.
    """

    recipe_id: int
    text: str

    @property
    def primary_id(self) -> int:
        return INFO_SECTION_CHUNK_ID_BASE - self.recipe_id

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "_InfoSection | None":
        if not recipe.info:
            return None
        rendered = "\n\n".join(
            _render_info_entry(entry) for entry in recipe.info
        ).strip()
        if not rendered:
            return None
        return cls(recipe_id=recipe.recipe_id, text=rendered)


def _render_info_entry(entry: dict[str, Any]) -> str:
    """Render one info entry (group_name + content) the same way the chunker does."""
    name = entry.get("group_name")
    content = entry.get("content", "") or ""
    if name:
        return f"## {name}\n{content}"
    return content


def _build_info_sections(recipes: list[Recipe]) -> list[_InfoSection]:
    return [s for s in (_InfoSection.from_recipe(r) for r in recipes) if s is not None]


def nearest_neighbour_ids(
    target_text: str,
    target_recipe_id: int,
    pool: list[Chunk],
    encoder: Encoder,
    k: int,
) -> list[int]:
    """Return up to ``k`` recipe_ids from ``pool`` whose chunks are most similar
    to ``target_text``. The pool's own recipe is excluded."""
    if not pool:
        return []
    pool = [c for c in pool if c.recipe_id != target_recipe_id]
    if not pool:
        return []
    target_vec = encoder.encode_query(target_text)
    pool_texts = [c.text for c in pool]
    pool_vecs = encoder._encode_documents(pool_texts)
    scores = pool_vecs @ target_vec
    order = scores.argsort()[::-1]
    ranked_recipes: list[int] = []
    seen: set[int] = set()
    for idx in order:
        rid = pool[idx].recipe_id
        if rid in seen:
            continue
        seen.add(rid)
        ranked_recipes.append(rid)
        if len(ranked_recipes) >= k:
            break
    return ranked_recipes


def _recipe_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    """Render a whole slice using the same shaper the chunker uses for whole chunks."""
    for c in _build_whole_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _ingredient_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _build_ingredient_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _instruction_text(recipes_by_id: dict[int, Recipe], recipe_id: int) -> str:
    for c in _build_instruction_chunks([recipes_by_id[recipe_id]]):
        return c.text
    raise KeyError(f"recipe {recipe_id} not found")


def _info_section_text(sections_by_primary: dict[int, _InfoSection], primary_id: int) -> str:
    section = sections_by_primary.get(primary_id)
    if section is None:
        raise KeyError(f"info section {primary_id} not found")
    return section.text


def selections_for_type(
    query_type: str,
    per_type_count: int,
    recipes: list[Recipe],
    encoder: Encoder,
    rng: random.Random,
) -> list[Selection]:
    """Return the selections (one per row) for a single query type."""
    recipes_by_id = {r.recipe_id: r for r in recipes}

    if query_type in {"recipe", "technique"}:
        # Both sample whole-recipe chunks and ask the model to find a notable
        # angle (a recipe, or a technique in a recipe). The selection shape is
        # identical; the prompt template is the only thing that differs.
        chunk_pool = _build_whole_chunks(recipes)
        picks = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        return [
            Selection(query_type=query_type, primary_id=c.chunk_id,
                       primary_text=c.text, primary_recipe_id=c.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for c in picks
        ]

    if query_type == "info":
        sections = _build_info_sections(recipes)
        picks = rng.sample(sections, k=min(per_type_count, len(sections)))
        return [
            Selection(query_type=query_type, primary_id=s.primary_id,
                       primary_text=s.text, primary_recipe_id=s.recipe_id,
                       sibling_texts=[], sibling_recipe_ids=[])
            for s in picks
        ]

    if query_type == "recipe_with_a_swap":
        chunk_pool = _build_whole_chunks(recipes)
        seeds = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
        selections: list[Selection] = []
        for seed in seeds:
            siblings = nearest_neighbour_ids(
                target_text=seed.text,
                target_recipe_id=seed.recipe_id,
                pool=chunk_pool,
                encoder=encoder,
                k=2,
            )
            if len(siblings) < 2:
                continue
            # Closest sibling is the swap target; the next-closest sibling is
            # the reference. The model only writes the prompt.
            swap_recipe_id, reference_recipe_id = siblings[0], siblings[1]
            selections.append(Selection(
                query_type=query_type,
                primary_id=seed.chunk_id,
                primary_text=seed.text,
                primary_recipe_id=seed.recipe_id,
                reference_text=_recipe_text(recipes_by_id, reference_recipe_id),
                swap_text=_recipe_text(recipes_by_id, swap_recipe_id),
                sibling_recipe_ids=[reference_recipe_id, swap_recipe_id],
            ))
        return selections

    if query_type in {
        "recipe_variation", "recipe_combination", "recipe_from_constraints",
        "recipe_adaptation_method",
    }:
        # All four create a NEW recipe from whole-recipe references. The
        # primary is the base recipe; the single sibling is the reference the
        # answer should draw from. Nearest-neighbour selection keeps the
        # reference related enough to adapt, fuse, or assemble well.
        chunk_pool = _build_whole_chunks(recipes)
        pool_to_recipe = {c.chunk_id: c.recipe_id for c in chunk_pool}
        return _suggestion_selections(
            query_type=query_type,
            per_type_count=per_type_count,
            encoder=encoder,
            rng=rng,
            chunk_id_to_recipe_id=pool_to_recipe,
            chunk_pool=chunk_pool,
            chunk_lookup=lambda rid: _recipe_text(recipes_by_id, rid),
            sibling_count=1,
        )

    if query_type == "recipe_suggestion_off_ingredients":
        chunk_pool = _build_ingredient_chunks(recipes)
        pool_to_recipe = {c.chunk_id: c.recipe_id for c in chunk_pool}
        return _suggestion_selections(
            query_type=query_type,
            per_type_count=per_type_count,
            encoder=encoder,
            rng=rng,
            chunk_id_to_recipe_id=pool_to_recipe,
            chunk_pool=chunk_pool,
            chunk_lookup=lambda rid: _ingredient_text(recipes_by_id, rid),
        )

    if query_type == "recipe_suggestion_off_instructions":
        chunk_pool = _build_instruction_chunks(recipes)
        pool_to_recipe = {c.chunk_id: c.recipe_id for c in chunk_pool}
        return _suggestion_selections(
            query_type=query_type,
            per_type_count=per_type_count,
            encoder=encoder,
            rng=rng,
            chunk_id_to_recipe_id=pool_to_recipe,
            chunk_pool=chunk_pool,
            chunk_lookup=lambda rid: _instruction_text(recipes_by_id, rid),
        )

    if query_type == "recipe_suggestion_off_info":
        sections = _build_info_sections(recipes)
        if len(sections) < per_type_count * 3:
            return []
        sections_by_primary = {s.primary_id: s for s in sections}
        sections_by_recipe = {s.recipe_id: s for s in sections}
        seeds = rng.sample(sections, k=min(per_type_count, len(sections)))
        selections = []
        for seed in seeds:
            siblings = nearest_neighbour_ids(
                target_text=seed.text,
                target_recipe_id=seed.recipe_id,
                pool=[_to_chunk(section) for section in sections],
                encoder=encoder,
                k=2,
            )
            if len(siblings) < 2:
                continue
            selections.append(Selection(
                query_type=query_type,
                primary_id=seed.primary_id,
                primary_text=seed.text,
                primary_recipe_id=seed.recipe_id,
                sibling_texts=[
                    _info_section_text(sections_by_primary, sections_by_recipe[s].primary_id)
                    for s in siblings
                ],
                sibling_recipe_ids=list(siblings),
            ))
        return selections

    raise ValueError(f"unknown query_type: {query_type}")


def _to_chunk(section: _InfoSection) -> Chunk:
    """Coerce an ``_InfoSection`` into a ``Chunk`` so it can flow through
    nearest-neighbour selection. Uses the same negative primary_id."""
    return Chunk(
        chunk_id=section.primary_id,
        recipe_id=section.recipe_id,
        chunk_type="info_section",
        text=section.text,
    )


def _suggestion_selections(
    query_type: str,
    per_type_count: int,
    encoder: Encoder,
    rng: random.Random,
    chunk_id_to_recipe_id: dict[int, int],
    chunk_pool: list[Chunk],
    chunk_lookup,
    sibling_count: int = 2,
) -> list[Selection]:
    """Shared selection logic for the multi-recipe query types.

    Each selection is 1 seed chunk + ``sibling_count`` nearest-neighbour chunks
    from distinct recipes, all from the same chunk-type pool. The pool's own
    ``chunk_id_to_recipe_id`` map is passed in because ids are stream-local.
    """
    if len(chunk_pool) < per_type_count * (sibling_count + 1):
        return []
    seeds = rng.sample(chunk_pool, k=min(per_type_count, len(chunk_pool)))
    selections: list[Selection] = []
    for seed in seeds:
        siblings = nearest_neighbour_ids(
            target_text=seed.text,
            target_recipe_id=seed.recipe_id,
            pool=chunk_pool,
            encoder=encoder,
            k=sibling_count,
        )
        if len(siblings) < sibling_count:
            continue
        selections.append(Selection(
            query_type=query_type,
            primary_id=seed.chunk_id,
            primary_text=seed.text,
            primary_recipe_id=chunk_id_to_recipe_id[seed.chunk_id],
            sibling_texts=[chunk_lookup(rid) for rid in siblings],
            sibling_recipe_ids=list(siblings),
        ))
    return selections
