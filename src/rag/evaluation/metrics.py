import math
from collections.abc import Sequence

def recall_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """Recall of the top-k: the fraction of the ground truth that is found.

    Counts how many ground-truth ids appear in the top-k and divides by the
    total number of ground-truth ids. The denominator is always the ground
    truth size, never k.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if len(ground_truth_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    retrieved_top_k = set(retrieved_ids[:k])
    ground_truths = set(ground_truth_ids)

    hits = len(retrieved_top_k.intersection(ground_truths))

    return hits / len(ground_truths)


def strict_recall_at_k(retrieved_ids: Sequence[int], ground_truth_ids: int | Sequence[int], k: int) -> float:
    """Recall at k accepting a single ground-truth id or a sequence.

    Convenience wrapper around recall_at_k for callers that hold one id
    rather than a list.
    """
    if isinstance(ground_truth_ids, int):
        ground_truth_ids = [ground_truth_ids]
    return recall_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=ground_truth_ids,
        k=k,
    )


def mrr_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """Reciprocal rank of the first relevant hit within the top-k.

    Scans the retrieved list in rank order and returns the reciprocal of the
    1-based rank of the first id that appears in the ground truth, or 0.0 when
    no ground-truth id appears in the top-k. This scores a single row; the
    mean across rows is applied by the caller.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if len(ground_truth_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    ground_truths = set(ground_truth_ids)
    for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        if retrieved_id in ground_truths:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """Precision of the top-k: the fraction of the window that is relevant.

    Counts how many ground-truth ids appear in the top-k and divides by k.
    The denominator is always k, so a retrieved list shorter than k counts
    the missing positions as non-relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if len(ground_truth_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    retrieved_top_k = set(retrieved_ids[:k])
    ground_truths = set(ground_truth_ids)
    hits = len(retrieved_top_k.intersection(ground_truths))
    return hits / k


def ndcg_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """Normalised discounted cumulative gain at k with binary relevance.

    Accumulates rel_i / log2(i+1) over the top-k, where rel_i is 1 when the
    recipe at rank i is in the ground truth, and divides by the ideal
    accumulation, which places every ground-truth id at the top. With binary
    relevance the standard DCG reduces to this simple sum.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if len(ground_truth_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    ground_truths = set(ground_truth_ids)
    dcg = 0.0
    for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        if retrieved_id in ground_truths:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(k, len(ground_truths))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg