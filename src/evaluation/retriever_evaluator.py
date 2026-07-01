""" 
Evaluation approach:

Recall@k:
    Creating an evaluation dataset is manually laborious.
    A LLM is thus used to generate queries and ground truths.

    However, some considerations:
        - Multiple queries have multiple ground truths.
            e.g. Queries: Chicken pasta?, Chicken alfredo?
                 Both can have different ground truths, which to a user is correct, but the system may penalize it.
        - LLM needs to generate user like queries.
        - What should the queries consider: recipe name, ingredients, instructions, ...
            Adding more information may make this more complicated and perhaps separate evaluations are needed for each.
    
    A clustering approach is used to group similar recipes together.
"""

from collections.abc import Sequence
from .metrics import recall_at_k, strict_recall_at_k
from src.models.retriever import Retriever
import numpy as np

def evaluate_strict(retriever: Retriever, evaluation_dataset: list[dict], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """ Evaluates a retriever on a dataset of queries and ground truths. """

    results = {k : [] for k in ks}
    max_k = max(ks)

    for sample in evaluation_dataset:
        query = sample["query"]
        source_recipe = sample["recipe_id"]

        retrieved_ids = retriever.retrieve(query, max_k)

        for k in ks:
            score = strict_recall_at_k(retrieved_ids, source_recipe, k)
            results[k].append(score)
    
    return {
        f"strict_recall@{k}": np.mean(results[k]) for k in ks
    }
