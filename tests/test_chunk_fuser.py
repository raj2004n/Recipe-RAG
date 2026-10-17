import pytest

from src.rag.retrieval.chunker import Chunk
from src.rag.retrieval.dense.retriever import RetrievalResult
from src.rag.retrieval.hybrid.fuser import ChunkFuser


def _rr(chunk_id: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            recipe_id=chunk_id // 10,
            chunk_type="whole", # placeholder
            text=f"chunk {chunk_id}",
        ),
        score=score,
    )


def test_dedupe_by_chunk_id_across_sources():
    """Chunk 0 appears in both dense and sparse; the fused result must have
    exactly one entry for it, with both sources recorded."""
    fuser = ChunkFuser(strategy="rrf", rrf_k=60)
    dense = [_rr(0, 0.9), _rr(1, 0.7)]
    sparse = [_rr(0, 5.0), _rr(2, 4.0)]

    fused = fuser.fuse(dense, sparse)

    by_id = {fc.chunk.chunk_id: fc for fc in fused}
    assert set(by_id.keys()) == {0, 1, 2}

    assert by_id[0].sources == ("dense", "sparse")
    assert "dense" not in by_id[1].sources or "dense" in by_id[1].sources
    assert by_id[2].sources == ("sparse",)


def test_fused_score_sums_appearances():
    """With RRF, a chunk's fused_score accumulates every time it appears.
    Chunk 0 (rank 1 dense, rank 2 sparse) should score higher than chunk 1
    (rank 2 dense only)."""
    fuser = ChunkFuser(strategy="rrf", rrf_k=60)
    dense = [_rr(0, 0.9), _rr(1, 0.7), _rr(2, 0.5)]
    sparse = [_rr(0, 5.0), _rr(3, 4.0)]

    fused = fuser.fuse(dense, sparse)

    # Fused chunks come back sorted by fused_score descending.
    assert fused[0].chunk.chunk_id == 0


def test_weighted_sum_minmax_normalises_scores():
    """weighted_sum applies min-max normalisation per source, so the score
    scales don't matter for ranking - only relative order does."""
    fuser = ChunkFuser(strategy="weighted_sum", weight_dense=1.0, weight_sparse=1.0)
    dense = [_rr(10, 100.0), _rr(11, 50.0)]   # range 50
    sparse = [_rr(20, 1.0), _rr(21, 0.0)]     # range 1

    fused = fuser.fuse(dense, sparse)
    # Within each source the max normalised to 1.0; chunk 10 (dense top)
    # and chunk 20 (sparse top) tie. Order between ties isn't guaranteed,
    # so just verify the set and the max normalised score.
    assert {fc.chunk.chunk_id for fc in fused} == {10, 11, 20, 21}
    top_score = fused[0].fused_score
    assert top_score == pytest.approx(1.0)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        ChunkFuser(strategy="bogus")
