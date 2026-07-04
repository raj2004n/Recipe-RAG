from collections.abc import Sequence

def recall_at_k(retrieved_ids: Sequence[int], ground_truth_ids: Sequence[int], k: int) -> float:
    """ Calculates recall@k for a list of ground truths and predictions."""
    # catch edge cases
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if len(ground_truth_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    retrieved_top_k = set(retrieved_ids[:k])
    ground_truths = set(ground_truth_ids)

    hits = len(retrieved_top_k.intersection(ground_truths))
    
    return hits / len(ground_truths)


def strict_recall_at_k(retrieved_ids: Sequence[int], ground_truth_id: int, k: int) -> float:
    """ Wrapper for strict recal_at_k."""
    return recall_at_k(
        retrieved_ids=retrieved_ids,
        ground_truth_ids=[ground_truth_id],
        k=k,
    )