import pytest

from src.rag.evaluation.metrics import (
    recall_at_k,
    mrr_at_k,
    precision_at_k,
    ndcg_at_k,
)

GT = [61, 232]
RETRIEVED = [191, 61, 232, 82, 676]


def test_worked_example_matches_scoring_logic():
    """The four metrics at k=1,2,3,5 on one retrieval, as documented."""
    assert recall_at_k(RETRIEVED, GT, 1) == 0.0
    assert mrr_at_k(RETRIEVED, GT, 1) == 0.0
    assert precision_at_k(RETRIEVED, GT, 1) == 0.0
    assert ndcg_at_k(RETRIEVED, GT, 1) == 0.0

    assert recall_at_k(RETRIEVED, GT, 2) == pytest.approx(0.5)
    assert mrr_at_k(RETRIEVED, GT, 2) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, GT, 2) == pytest.approx(0.5)
    assert ndcg_at_k(RETRIEVED, GT, 2) == pytest.approx(0.387, abs=0.0005)

    assert recall_at_k(RETRIEVED, GT, 3) == pytest.approx(1.0)
    assert mrr_at_k(RETRIEVED, GT, 3) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, GT, 3) == pytest.approx(0.667, abs=0.0005)
    assert ndcg_at_k(RETRIEVED, GT, 3) == pytest.approx(0.693, abs=0.0005)

    assert recall_at_k(RETRIEVED, GT, 5) == pytest.approx(1.0)
    assert mrr_at_k(RETRIEVED, GT, 5) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, GT, 5) == pytest.approx(0.4)
    assert ndcg_at_k(RETRIEVED, GT, 5) == pytest.approx(0.693, abs=0.0005)


def test_mrr_uses_first_relevant_rank():
    """MRR reflects the first relevant hit only, not the best or the mean."""
    assert mrr_at_k([999, 61, 232], GT, 3) == pytest.approx(0.5)
    assert mrr_at_k([61, 999, 232], GT, 3) == pytest.approx(1.0)


def test_no_relevant_hit_scores_zero():
    for metric in (recall_at_k, mrr_at_k, precision_at_k, ndcg_at_k):
        assert metric([999, 998], GT, 2) == 0.0


def test_short_retrieved_list_keeps_k_denominator_for_precision():
    """A retrieved list shorter than k scores over what is there; precision
    still divides by k, so missing positions count as non-relevant."""
    assert precision_at_k([61], GT, 5) == pytest.approx(0.2)
    assert recall_at_k([61], GT, 5) == pytest.approx(0.5)


def test_ndcg_saturates_once_all_ground_truth_is_in_window():
    """With |GT| < k, the ideal gain is fixed, so nDCG stops rising once
    every ground-truth id is in the window."""
    assert ndcg_at_k(RETRIEVED, GT, 3) == pytest.approx(ndcg_at_k(RETRIEVED, GT, 5))


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, precision_at_k, ndcg_at_k])
def test_k_must_be_positive(metric):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        metric(RETRIEVED, GT, 0)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, precision_at_k, ndcg_at_k])
def test_empty_ground_truth_raises(metric):
    with pytest.raises(ValueError, match="relevant_ids cannot be empty"):
        metric(RETRIEVED, [], 3)
