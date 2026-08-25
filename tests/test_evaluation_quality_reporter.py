"""Tests for the dataset-quality reporter.

All fixtures are synthetic and live under tmp_path. The reporter's contract is
"two datasets reported separately, never blended" - the tests below exercise
each metric in isolation against that contract.
"""

import json
from pathlib import Path

from src.rag.evaluation.quality_reporter import (
    DatasetQuality,
    format_table,
    report_quality,
    write_json,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write a JSONL file with one object per line."""
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _write_corpus(path: Path, n: int) -> Path:
    """Write a synthetic recipes corpus with `n` placeholder recipes."""
    recipes = [{"recipe_id": i, "name": f"recipe {i}"} for i in range(n)]
    path.write_text(json.dumps(recipes))
    return path


def test_reports_rows_and_distinct_queries(tmp_path):
    """Rows, distinct queries, query diversity, distinct recipes, avg per recipe."""
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "chicken pasta", "recipe_ids": [1]},
        {"query": "chicken pasta", "recipe_ids": [2]},
        {"query": "vegan tacos", "recipe_ids": [3]},
        {"query": "vegan tacos", "recipe_ids": [3]},
        {"query": "lamb curry", "recipe_ids": [5]},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 10)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.rows == 5
    assert report.golden.distinct_queries == 3
    assert report.golden.query_diversity == 3 / 5
    assert report.golden.distinct_recipes == 4
    assert report.golden.avg_queries_per_recipe == 5 / 4


def test_coverage_against_corpus_size(tmp_path):
    """Coverage divides by corpus size, not by the number of referenced recipes."""
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "a", "recipe_ids": [1]},
        {"query": "b", "recipe_ids": [2]},
        {"query": "c", "recipe_ids": [3]},
        {"query": "d", "recipe_ids": [4]},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 10)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.coverage == 0.4


def test_ambiguity_per_query_definition_b(tmp_path):
    """Two rows for the same query pointing at different recipes → ambiguous.

    The per-query definition (definition B) counts distinct *queries* that map
    to more than one distinct recipe. A query with one row pointing at one
    recipe is not ambiguous, regardless of how many rows reference that recipe.
    """
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "chicken pasta", "recipe_ids": [42]},
        {"query": "chicken pasta", "recipe_ids": [137]},
        {"query": "tacos", "recipe_ids": [88]},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 200)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.ambiguity_rate == 0.5


def test_verified_and_reference_answer_rates(tmp_path):
    """Half the rows are verified with a reference answer; half are not."""
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "a", "recipe_ids": [1], "verified": True, "reference_answer": "Use a wok."},
        {"query": "b", "recipe_ids": [2], "verified": True, "reference_answer": "Marinate overnight."},
        {"query": "c", "recipe_ids": [3], "verified": False, "reference_answer": None},
        {"query": "d", "recipe_ids": [4], "verified": False, "reference_answer": ""},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 10)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.verified_rate == 0.5
    assert report.golden.reference_answer_rate == 0.5


def test_source_breakdown(tmp_path):
    """Three sources plus a missing-source bucket; counts must sum to rows."""
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "a", "recipe_ids": [1], "source": "llm-draft"},
        {"query": "b", "recipe_ids": [2], "source": "llm-draft"},
        {"query": "c", "recipe_ids": [3], "source": "llm-draft"},
        {"query": "d", "recipe_ids": [4], "source": "human"},
        {"query": "e", "recipe_ids": [5], "source": "human"},
        {"query": "f", "recipe_ids": [6]},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 10)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.source_breakdown == {"llm-draft": 3, "human": 2, "unknown": 1}
    assert sum(report.golden.source_breakdown.values()) == report.golden.rows


def test_empty_file_does_not_crash(tmp_path):
    """Zero-byte files produce zero-row reports with non-null warnings."""
    (tmp_path / "golden.jsonl").write_text("")
    (tmp_path / "llm.jsonl").write_text("")
    corpus = _write_corpus(tmp_path / "corpus.json", 5)
    report = report_quality(str(tmp_path / "golden.jsonl"), str(tmp_path / "llm.jsonl"), str(corpus))
    assert report.golden.rows == 0
    assert report.llm_generated.rows == 0
    assert report.golden.warning is not None and "empty" in report.golden.warning
    assert report.llm_generated.warning is not None and "empty" in report.llm_generated.warning
    assert report.golden.coverage == 0.0
    assert report.llm_generated.coverage == 0.0


def test_missing_file_does_not_crash(tmp_path):
    """Absent files produce zero-row reports with 'file not found' warnings."""
    corpus = _write_corpus(tmp_path / "corpus.json", 5)
    report = report_quality(
        str(tmp_path / "no_golden.jsonl"),
        str(tmp_path / "no_llm.jsonl"),
        str(corpus),
    )
    assert report.golden.warning is not None and "not found" in report.golden.warning
    assert report.llm_generated.warning is not None and "not found" in report.llm_generated.warning


def test_legacy_rows_are_counted_and_surfaced_in_warning(tmp_path):
    """Legacy `recipe_id` rows are counted and folded into the warning string."""
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "a", "recipe_ids": [1]},
        {"query": "b", "recipe_ids": [2]},
        {"query": "c", "recipe_ids": [3]},
        {"query": "d", "recipe_id": 4},
        {"query": "e", "recipe_id": 5},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 10)
    report = report_quality(str(golden), str(tmp_path / "missing.jsonl"), str(corpus))
    assert report.golden.legacy_rows == 2
    assert report.golden.warning is not None
    assert "2 row(s) used legacy recipe_id field" in report.golden.warning
    # New-schema rows still resolve correctly despite mixed-schema file.
    assert report.golden.distinct_recipes == 5


def test_cli_prints_both_tables_and_exits_zero(tmp_path, monkeypatch, capsys):
    """End-to-end CLI run: synthetic corpus + eval files, monkeypatched DATA_PATHS.

    We call run() in-process instead of spawning a subprocess: the script reads
    DATA_PATHS at call time, so monkeypatching works, and the test stays fast
    and side-effect-free. The script writes its JSON output next to the
    golden file, so we read from there rather than from tmp_path's root.
    """
    golden = _write_jsonl(tmp_path / "golden.jsonl", [
        {"query": "a", "recipe_ids": [1], "source": "human", "verified": True, "reference_answer": "x"},
    ])
    llm = _write_jsonl(tmp_path / "llm.jsonl", [
        {"query": "b", "recipe_ids": [2], "source": "llm-draft"},
    ])
    corpus = _write_corpus(tmp_path / "corpus.json", 5)

    import scripts.report_dataset_quality as cli
    monkeypatch.setitem(cli.DATA_PATHS, "eval_golden", str(golden))
    monkeypatch.setitem(cli.DATA_PATHS, "eval_llm_generated", str(llm))
    monkeypatch.setitem(cli.DATA_PATHS, "eval_corpus_recipes", str(corpus))

    assert cli.run() == 0

    captured = capsys.readouterr()
    assert "golden" in captured.out
    assert "llm_generated" in captured.out
    assert "Eval dataset quality report" in captured.out

    report_json = golden.parent / ".quality_report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text())
    assert payload["corpus_size"] == 5
    assert payload["golden"]["rows"] == 1
    assert payload["llm_generated"]["rows"] == 1
