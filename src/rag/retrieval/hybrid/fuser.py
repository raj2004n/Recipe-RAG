"""Fuse dense and sparse retrieval results into a single ranked list.

The fuser handles two jobs at once:
  1. Merge two `list[RetrievalResult]` (different score scales) into one.
  2. Dedupe by `chunk.chunk_id` so the reranker never sees the same chunk twice.

Supported strategies:
  - "rrf":          Reciprocal Rank Fusion. score = w / (k + rank).
                    Score-scale agnostic; one knob (`rrf_k`).
  - "weighted_sum": Min-max normalise each source's scores to [0, 1], then
                    score = w_d * norm_d + w_s * norm_s. Two knobs
                    (weight_dense, weight_sparse). Brittle if one side's
                    distribution shifts.
"""

from dataclasses import dataclass, field
from typing import Iterable

from ..dense.retriever import RetrievalResult


@dataclass
class FusedChunk:
    """A chunk after fusion + dedupe; ready for the reranker."""

    chunk: object  # chunker.Chunk - kept untyped to avoid circular import
    fused_score: float
    sources: tuple[str, ...] = field(default_factory=tuple)


class ChunkFuser:
    """Fuse dense + sparse `RetrievalResult` lists into ranked `FusedChunk`s."""

    STRATEGIES = ("rrf", "weighted_sum")

    def __init__(
        self,
        strategy: str = "rrf",
        rrf_k: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy {strategy!r}; "
                f"choose from {self.STRATEGIES}"
            )
        self.strategy = strategy
        self.rrf_k = rrf_k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse


    def fuse(
        self,
        dense: list[RetrievalResult],
        sparse: list[RetrievalResult],
    ) -> list[FusedChunk]:
        """Merge the two lists and dedupe by `chunk.chunk_id`."""
        if self.strategy == "rrf":
            return self._fuse_rrf(dense, sparse)
        return self._fuse_weighted_sum(dense, sparse)


    def _fuse_rrf(
        self,
        dense: list[RetrievalResult],
        sparse: list[RetrievalResult],
    ) -> list[FusedChunk]:
        """Reciprocal Rank Fusion; ranks are 1-indexed."""
        by_id: dict[int, FusedChunk] = {}

        for rank, result in enumerate(_ranked(dense), start=1):
            score = self.weight_dense / (self.rrf_k + rank)
            self._accumulate(by_id, result, score, source="dense")

        for rank, result in enumerate(_ranked(sparse), start=1):
            score = self.weight_sparse / (self.rrf_k + rank)
            self._accumulate(by_id, result, score, source="sparse")

        return sorted(
            by_id.values(),
            key=lambda fc: fc.fused_score,
            reverse=True,
        )


    def _fuse_weighted_sum(
        self,
        dense: list[RetrievalResult],
        sparse: list[RetrievalResult],
    ) -> list[FusedChunk]:
        """Min-max normalise each source then weighted-sum. RRF is recommended."""
        by_id: dict[int, FusedChunk] = {}

        for result, norm in zip(dense, _minmax(dense)):
            score = self.weight_dense * norm
            self._accumulate(by_id, result, score, source="dense")

        for result, norm in zip(sparse, _minmax(sparse)):
            score = self.weight_sparse * norm
            self._accumulate(by_id, result, score, source="sparse")

        return sorted(
            by_id.values(),
            key=lambda fc: fc.fused_score,
            reverse=True,
        )


    @staticmethod
    def _accumulate(
        by_id: dict[int, FusedChunk],
        result: RetrievalResult,
        score: float,
        source: str,
    ) -> None:
        chunk_id = result.chunk.chunk_id
        existing = by_id.get(chunk_id)
        if existing is None:
            by_id[chunk_id] = FusedChunk(
                chunk=result.chunk,
                fused_score=score,
                sources=(source,),
            )
        else:
            existing.fused_score += score
            if source not in existing.sources:
                existing.sources = existing.sources + (source,)


def _ranked(results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
    """Sort by score desc so RRF gets a consistent rank order."""
    return sorted(results, key=lambda r: r.score, reverse=True)


def _minmax(results: list[RetrievalResult]) -> list[float]:
    """Map scores to [0, 1] preserving order. Constant-score lists become all 1.0."""
    if not results:
        return []
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]
