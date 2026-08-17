from typing import List
from .bm25indexer import BM25IndexedChunkSet
from .tokenizer import Tokenizer
from ..dense.retriever import RetrievalResult

class BM25Retriever:
    def __init__(self, indexed_chunk_set: BM25IndexedChunkSet, tokenizer: Tokenizer):
        self.indexed_chunk_set = indexed_chunk_set
        self.tokenizer = tokenizer

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        query_tokens = self.tokenizer.tokenize(query)

        scores = self.indexed_chunk_set.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for idx in ranked_indices[:k]:
            score = scores[idx]

            if score <= 0:
                continue

            chunk = self.indexed_chunk_set.tokenized_chunks[idx].chunk

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        return results