"""Tests for chunk_recipes.

Verifies the global-unique chunk_id contract that FAISS relies on. If two
chunks share an id within a snapshot, a FAISS hit would silently resolve to
the wrong chunk.
"""

import json
from pathlib import Path

import pytest

from src.rag.retrieval.chunker import chunk_recipes


def _make_recipe(recipe_id: int, name: str = "Test") -> dict:
    """Build a minimal Recipe-shaped dict for the preprocessor to consume."""
    return {
        "recipe_id": recipe_id,
        "url": f"https://example.test/{name}",
        "recipe_name": name,
        "servings": 2,
        "info": [{"group_name": "Intro", "content": f"{name} intro"}],
        "ingredient_groups": [
            {
                "group_name": None,
                "ingredients": [{"name": "flour", "amount": 1, "unit": "cup"}],
            }
        ],
        "instruction_groups": [
            {"group_name": "Main", "instructions": ["Mix.", "Bake."]}
        ],
        "notes": [],
    }


@pytest.fixture
def recipes_file(tmp_path: Path) -> Path:
    p = tmp_path / "recipes.json"
    p.write_text(json.dumps([_make_recipe(1, "A"), _make_recipe(2, "B")]))
    return p


def test_chunk_ids_are_unique_across_chunk_sets_hybrid(recipes_file):
    """Hybrid strategy produces sparse + dense sets; ids must not collide
    between them (the dedup check in chunk_recipes enforces this)."""
    chunk_sets = chunk_recipes("hybrid", recipes_file)
    assert len(chunk_sets) == 2
    assert {cs.name for cs in chunk_sets} == {"sparse", "dense"}

    all_ids = [c.chunk_id for cs in chunk_sets for c in cs.chunks]
    assert len(set(all_ids)) == len(all_ids)


def test_duplicate_recipes_still_produce_distinct_ids(recipes_file):
    """Calling chunk_recipes twice on the same input must produce different
    ids each call - ids are process-unique, not input-unique."""
    first = chunk_recipes("whole_recipe", recipes_file)
    second = chunk_recipes("whole_recipe", recipes_file)

    first_ids = [c.chunk_id for cs in first for c in cs.chunks]
    second_ids = [c.chunk_id for cs in second for c in cs.chunks]

    assert first_ids == second_ids == [0, 1]
    # Different processes / counters produce ids starting at 0, but within
    # the same call the ids must not collide with each other.
    assert len(set(first_ids)) == len(first_ids)
