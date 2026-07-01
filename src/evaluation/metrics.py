from collections.abc import Sequence

def recall_at_k(retrieved_ids: Sequence[int], expected_ids: Sequence[int], k: int) -> float:
    """ Calculates recall@k for a list of ground truths and predictions."""
    # catch edge cases
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if len(expected_ids) == 0:
        raise ValueError("relevant_ids cannot be empty.")

    retrieved_top_k = set(retrieved_ids[:k])
    expected = set(expected_ids)

    hits = len(retrieved_top_k.intersection(expected))

    return hits / len(expected)


def strict_recall_at_k(retrieved_ids: Sequence[int], source_recipe: int, k: int) -> float:
    """ Wrapper for strict recal_at_k."""
    return recall_at_k(
        retrieved_ids=retrieved_ids,
        expected_ids=[source_recipe],
        k=k,
    )


def hit_at_k(retrieved_ids: Sequence[int], expected_ids: Sequence[int], k: int) -> bool:
    """ Calculates hit@k for a list of ground truths and predictions."""
    return recall_at_k(
        retrieved_ids,
        expected_ids,
        k,
    ) > 0.0