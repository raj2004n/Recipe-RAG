"""Pipeline-agnostic retrieval evaluation.

The evaluator scores a retriever against eval rows of (query, recipe_ids)
ground truth. It is agnostic to the retrieval implementation: it takes a
retrieve(query, k) -> list[int] callable that returns recipe_ids in rank
order, so both the dense and hybrid pipelines are scored through the same
code path.
"""

from collections.abc import Callable, Sequence
from .metrics import recall_at_k, mrr_at_k, precision_at_k, ndcg_at_k

_METRICS = (
    ("recall", recall_at_k),
    ("mrr", mrr_at_k),
    ("precision", precision_at_k),
    ("ndcg", ndcg_at_k),
)


class RetrievalEvaluator:
    def __init__(self, retrieve: Callable[[str, int], Sequence[int]]):
        self._retrieve = retrieve

    def evaluate(self, rows: Sequence[dict], ks: Sequence[int] = (1, 2, 3, 5, 10)) -> dict:
        """Score every row and aggregate over the slice and per query type.

        Each row is scored from a single retrieval call at the largest k, so
        all metrics at all k values describe the same ranked list. The slice
        aggregate is the mean over the given rows; the query-type aggregates
        are the means over the rows of each query_type. Rows from different
        slices must be evaluated separately so the slices are never blended.
        """
        if not ks:
            raise ValueError("ks must not be empty.")
        if any(k <= 0 for k in ks):
            raise ValueError("every k must be greater than zero.")

        max_k = max(ks)
        per_row = []
        for row in rows:
            query = row["query"]
            ground_truth = list(row["recipe_ids"])
            retrieved = list(self._retrieve(query, max_k))

            scores = {}
            for k in ks:
                for name, metric in _METRICS:
                    scores[f"{name}@{k}"] = metric(retrieved, ground_truth, k)

            per_row.append(
                {
                    "query": query,
                    "query_type": row.get("query_type"),
                    "recipe_ids": ground_truth,
                    "retrieved_ids": retrieved,
                    "scores": scores,
                }
            )

        return {
            "per_row": per_row,
            "slice": _mean_scores(per_row),
            "query_type": {
                query_type: _mean_scores(
                    [r for r in per_row if r["query_type"] == query_type]
                )
                for query_type in sorted({r["query_type"] for r in per_row})
            },
        }


def _mean_scores(rows: Sequence[dict]) -> dict[str, float]:
    """Mean of the per-row score dicts, keyed by metric@k."""
    if not rows:
        return {}
    keys = rows[0]["scores"].keys()
    return {
        key: sum(r["scores"][key] for r in rows) / len(rows)
        for key in keys
    }
