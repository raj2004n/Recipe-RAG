"""Tests for the PR 3 golden-subset tooling: drafter, export, apply."""

import json
from pathlib import Path

from src.rag.data.golden_drafter import (
    DraftList,
    DraftRecord,
    _ingredient_list,
    _instruction_text,
    _validate_query_type,
    build_prompt,
    load_processed_ids,
)
import scripts.apply_decisions as apply_cli
import scripts.export_candidates_for_review as export_cli


# ---------- drafter helpers ----------


def test_ingredient_list_flattens_groups():
    """Multiple ingredient groups collapse into a single flat list."""
    recipe = {
        "ingredient_groups": [
            {"ingredients": [{"name": "flour"}, {"name": "sugar"}]},
            {"ingredients": [{"name": "butter"}]},
        ]
    }
    assert _ingredient_list(recipe) == ["flour", "sugar", "butter"]


def test_ingredient_list_skips_unnamed():
    """Ingredients with no name are silently dropped."""
    recipe = {"ingredient_groups": [{"ingredients": [{"name": "flour"}, {"amount": 1}]}]}
    assert _ingredient_list(recipe) == ["flour"]


def test_instruction_text_flattens_groups():
    """Instruction steps join into a single newline-delimited string."""
    recipe = {"instruction_groups": [{"instructions": ["step 1", "step 2"]}, {"instructions": ["step 3"]}]}
    assert _instruction_text(recipe) == "step 1\nstep 2\nstep 3"


def test_validate_query_type_clamps_unknown():
    """query_type is bucketed into one of four known values, else 'unknown'."""
    assert _validate_query_type("ingredients") == "ingredients"
    assert _validate_query_type("technique") == "technique"
    assert _validate_query_type("name") == "name"
    assert _validate_query_type("dietary") == "dietary"
    assert _validate_query_type("something else") == "unknown"
    assert _validate_query_type("") == "unknown"


def test_build_prompt_includes_recipe_text():
    """The prompt must contain the recipe name, ingredients, and instructions."""
    recipe = {
        "recipe_name": "Chicken Pasta",
        "ingredient_groups": [{"ingredients": [{"name": "chicken"}, {"name": "pasta"}]}],
        "instruction_groups": [{"instructions": ["cook chicken", "boil pasta"]}],
    }
    prompt = build_prompt(recipe)
    assert "Chicken Pasta" in prompt
    assert "chicken" in prompt
    assert "pasta" in prompt
    assert "cook chicken" in prompt
    assert "boil pasta" in prompt


def test_build_prompt_tells_model_to_extract():
    """Prompt must include the extract-don't-invent guard."""
    recipe = {"recipe_name": "x", "ingredient_groups": [], "instruction_groups": []}
    prompt = build_prompt(recipe)
    assert "extract" in prompt.lower() or "use only" in prompt.lower()


def test_load_processed_ids_resumes():
    """load_processed_ids reads recipe_ids and falls back to recipe_id."""
    path = Path("/tmp/_test_processed.jsonl")
    rows = [
        {"recipe_ids": [1, 2]},
        {"recipe_ids": [3]},
        {"recipe_id": 4},
        {"recipe_id": 5},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    try:
        processed = load_processed_ids(path)
        assert processed == {1, 2, 3, 4, 5}
    finally:
        path.unlink()


def test_draft_record_and_list_round_trip():
    """DraftRecord / DraftList parse what the model is expected to emit."""
    payload = json.dumps({
        "candidates": [
            {"query": "q1", "reference_answer": "a1", "query_type": "ingredients"},
            {"query": "q2", "reference_answer": "a2", "query_type": "technique"},
        ]
    })
    parsed = DraftList.model_validate_json(payload)
    assert len(parsed.candidates) == 2
    assert parsed.candidates[0].query == "q1"
    assert isinstance(parsed.candidates[0], DraftRecord)


# ---------- export CLI ----------


def _write_recipes_and_candidates(tmp_path):
    """Helper: build a tiny corpus + candidates file."""
    corpus = [
        {"recipe_id": 1, "recipe_name": "Chicken pasta", "ingredient_groups": [{"ingredients": [{"name": "chicken"}, {"name": "pasta"}]}]},
        {"recipe_id": 2, "recipe_name": "Beef tacos", "ingredient_groups": [{"ingredients": [{"name": "beef"}, {"name": "tortilla"}]}]},
    ]
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    candidates = []
    for i in range(1, 12):  # 11 candidates so the batch-cap kicks in
        candidates.append({
            "query": f"query {i}",
            "recipe_ids": [((i - 1) % 2) + 1],
            "reference_answer": f"ref {i}",
            "query_type": "ingredients",
            "source": "llm-draft",
            "verified": False,
        })
    cand_path = tmp_path / "candidates.jsonl"
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n")
    return cand_path


def test_export_writes_at_most_ten_per_batch(tmp_path, monkeypatch):
    """A 11-candidate file emits a batch with 10 rows; remaining rows go to next batch."""
    cand_path = _write_recipes_and_candidates(tmp_path)
    import argparse
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "cleaned_recipes", str(tmp_path / "corpus.json"))
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "eval_golden_candidates", str(cand_path))

    rc = export_cli.run(argparse.Namespace(candidates=None, batch_index=None))
    assert rc == 0
    html_path = tmp_path / ".candidates_review_batch_001.html"
    assert html_path.exists()
    html = html_path.read_text()
    # 10 candidate sections in the HTML
    assert html.count('cand #') == 10
    # The 11th row should not be in the first batch file
    assert "query 11" not in html


def test_export_html_contains_recipe_snippet(tmp_path, monkeypatch):
    """Each candidate card includes the linked recipe's name and an ingredient."""
    cand_path = _write_recipes_and_candidates(tmp_path)
    import argparse
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "cleaned_recipes", str(tmp_path / "corpus.json"))
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "eval_golden_candidates", str(cand_path))

    export_cli.run(argparse.Namespace(candidates=None, batch_index=None))
    html = (tmp_path / ".candidates_review_batch_001.html").read_text()
    assert "Chicken pasta" in html or "Beef tacos" in html


def test_export_handles_empty_candidates(tmp_path, monkeypatch):
    """Empty candidates file is a no-op (no HTML written, exit 0)."""
    cand_path = tmp_path / "candidates.jsonl"
    cand_path.write_text("")
    (tmp_path / "corpus.json").write_text("[]")
    import argparse
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "cleaned_recipes", str(tmp_path / "corpus.json"))
    monkeypatch.setitem(__import__("src.rag.core.config", fromlist=["DATA_PATHS"]).DATA_PATHS, "eval_golden_candidates", str(cand_path))

    rc = export_cli.run(argparse.Namespace(candidates=None, batch_index=None))
    assert rc == 0
    assert not (tmp_path / ".candidates_review_batch_001.html").exists()


# ---------- apply CLI ----------


def test_apply_decisions_round_trip(tmp_path):
    """correct / fix / drop / no-decision -> expected golden.jsonl contents."""
    candidates = [
        {"query": "q1", "recipe_ids": [1], "reference_answer": "r1", "query_type": "ingredients", "source": "llm-draft", "verified": False},
        {"query": "q2", "recipe_ids": [2], "reference_answer": "r2", "query_type": "technique", "source": "llm-draft", "verified": False},
        {"query": "q3", "recipe_ids": [3], "reference_answer": "r3", "query_type": "name", "source": "llm-draft", "verified": False},
        {"query": "q4", "recipe_ids": [4], "reference_answer": "r4", "query_type": "dietary", "source": "llm-draft", "verified": False},
    ]
    cand_path = tmp_path / "candidates.jsonl"
    cand_path.write_text("\n".join(json.dumps(c) for c in candidates) + "\n")

    journal = tmp_path / ".decisions_batch_001.jsonl"
    decisions = [
        {"candidate_index": 0, "decision": "correct", "corrections": {}},
        {"candidate_index": 1, "decision": "fix", "corrections": {"query": "q2 corrected", "reference_answer": "r2 corrected"}},
        {"candidate_index": 2, "decision": "drop", "corrections": {}},
        # index 3: no decision -> skipped
    ]
    journal.write_text("\n".join(json.dumps(d) for d in decisions) + "\n")

    golden_path = tmp_path / "golden.jsonl"
    rc = apply_cli.run(apply_cli.main.__globals__["argparse"].Namespace(candidates=str(cand_path), output=str(golden_path), journal=None))
    assert rc == 0
    out = [json.loads(line) for line in golden_path.read_text().splitlines() if line]
    assert len(out) == 2
    assert out[0]["verified"] is True
    assert out[0]["query"] == "q1"
    assert out[1]["verified"] is True
    assert out[1]["query"] == "q2 corrected"
    assert out[1]["reference_answer"] == "r2 corrected"


def test_apply_decisions_recipe_id_corrections(tmp_path):
    """Fixing recipe_ids accepts comma-separated input."""
    candidates = [
        {"query": "q1", "recipe_ids": [1], "reference_answer": "r1", "query_type": "ingredients", "source": "llm-draft", "verified": False},
    ]
    cand_path = tmp_path / "candidates.jsonl"
    cand_path.write_text(json.dumps(candidates[0]) + "\n")
    journal = tmp_path / ".decisions_batch_001.jsonl"
    journal.write_text(json.dumps({
        "candidate_index": 0,
        "decision": "fix",
        "corrections": {"recipe_ids": "1, 7, 12"},
    }) + "\n")

    golden_path = tmp_path / "golden.jsonl"
    apply_cli.run(apply_cli.main.__globals__["argparse"].Namespace(candidates=str(cand_path), output=str(golden_path), journal=None))
    row = json.loads(golden_path.read_text().strip())
    assert row["recipe_ids"] == [1, 7, 12]
    assert row["verified"] is True


def test_apply_decisions_no_journal_is_noop(tmp_path):
    """No journal file present -> empty golden.jsonl, exit 0."""
    candidates = [{"query": "q1", "recipe_ids": [1], "reference_answer": "r1", "query_type": "ingredients", "source": "llm-draft", "verified": False}]
    cand_path = tmp_path / "candidates.jsonl"
    cand_path.write_text(json.dumps(candidates[0]) + "\n")
    golden_path = tmp_path / "golden.jsonl"
    rc = apply_cli.run(apply_cli.main.__globals__["argparse"].Namespace(candidates=str(cand_path), output=str(golden_path), journal=None))
    assert rc == 0
    assert golden_path.read_text() == ""
