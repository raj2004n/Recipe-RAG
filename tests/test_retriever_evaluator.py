"""Tests for the pipeline-agnostic RetrievalEvaluator."""

import pytest

from src.rag.evaluation.retriever_evaluator import RetrievalEvaluator


def _fixed_retrieve(ranked):
    """A retrieve callable that returns a fixed ranked list, truncated to k."""
    def retrieve(query, k):
        return ranked[:k]
    return retrieve


def test_evaluate_returns_per_row_slice_and_query_type():
    rows = [
        {"query": "q1", "query_type": "recipe_variation", "recipe_ids": [61, 232]},
        {"query": "q2", "query_type": "recipe_variation", "recipe_ids": [191]},
        {"query": "q3", "query_type": "recipe", "recipe_ids": [999]},
    ]
    result = RetrievalEvaluator(_fixed_retrieve([191, 61, 232, 82, 676])).evaluate(
        rows, ks=(1, 2, 3)
    )

    # Per-row: q1 matches the worked example.
    q1 = result["per_row"][0]
    assert q1["retrieved_ids"] == [191, 61, 232]
    assert q1["scores"]["recall@3"] == pytest.approx(1.0)
    assert q1["scores"]["mrr@2"] == pytest.approx(0.5)

    # Slice mean over the three rows.
    assert result["slice"]["recall@1"] == pytest.approx(1 / 3)
    assert result["slice"]["recall@3"] == pytest.approx(2 / 3)

    # Query-type grouping: the no-hit row is isolated under its own type.
    assert result["query_type"]["recipe"]["recall@1"] == 0.0
    assert result["query_type"]["recipe_variation"]["recall@1"] == pytest.approx(0.5)


def test_one_retrieval_call_per_row_at_max_k():
    """Each row is scored from a single retrieval call at the largest k, so
    all metrics at all k values describe the same ranked list."""
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        return [1, 2, 3]

    rows = [
        {"query": "a", "recipe_ids": [1]},
        {"query": "b", "recipe_ids": [2]},
    ]
    RetrievalEvaluator(retrieve).evaluate(rows, ks=(1, 2, 3))
    assert calls == [("a", 3), ("b", 3)]


def test_empty_ks_raises():
    with pytest.raises(ValueError, match="ks must not be empty"):
        RetrievalEvaluator(lambda q, k: []).evaluate(
            [{"query": "a", "recipe_ids": [1]}], ks=()
        )
