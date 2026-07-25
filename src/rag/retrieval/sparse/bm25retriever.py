from typing import List
from .bm25indexer import BM25Index
from .tokenizer import Tokenizer
from ..dense.retriever import RetrievalResult

class BM25Retriever:
    def __init__(self, index: BM25Index, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        query_tokens = self.tokenizer.tokenize(query)
        scores = self.index.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for idx in ranked_indices[:k]:
            chunk = self.index.documents[idx]
            if scores[idx] <= 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(scores[idx])
                )
            )

        return results
    