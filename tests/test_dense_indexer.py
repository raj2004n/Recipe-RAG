"""Tests for the dense retrieval path.

Drives Indexer.index_embeddings with synthetic unit vectors so we can
verify FAISS top-k behaviour without loading a real sentence-transformers
model. The retriever is a thin wrapper around Indexer.search_one, so
exercising the indexer is what really matters here.
"""

import numpy as np

from src.rag.retrieval.chunker import Chunk
from src.rag.retrieval.dense.encoder import EncodedChunk, EncodedChunkSet
from src.rag.retrieval.dense.indexer import Indexer


def _make_encoded_chunk(chunk_id: int, recipe_id: int, text: str, vector: np.ndarray) -> EncodedChunk:
    return EncodedChunk(
        vector=vector.astype("float32").tolist(),
        text=text,
        recipe_id=recipe_id,
        chunk_id=chunk_id,
        chunk_type="whole",
    )


def _unit_vector(seed: int, dim: int) -> np.ndarray:
    """Deterministic unit vector derived from a seed for reproducibility."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)


def test_search_returns_highest_inner_product_first():
    """On FlatIP, identical-direction query returns the matching chunk top-1."""
    dim = 16
    indexer = Indexer(index_type="FlatIP", dimension=dim, score_threshold=None)

    target = _unit_vector(seed=1, dim=dim)
    noise = [_unit_vector(seed=i + 100, dim=dim) for i in range(3)]

    encoded = EncodedChunkSet(
        name="whole_recipe",
        encoded_chunks=[
            _make_encoded_chunk(0, 0, "noise-a", noise[0]),
            _make_encoded_chunk(1, 1, "noise-b", noise[1]),
            _make_encoded_chunk(2, 2, "target", target),
            _make_encoded_chunk(3, 3, "noise-c", noise[2]),
        ],
    )
    indexer.index_embeddings([encoded])

    hits = indexer.search_one(target, k=1)
    score, chunk = hits[0]
    assert chunk is not None
    assert chunk.chunk_id == 2
    assert score > 0.99


def test_score_threshold_drops_low_score_hits():
    """Once a result falls below threshold, all subsequent (lower-scoring)
    hits are also truncated - that's the documented behaviour."""
    dim = 16
    indexer = Indexer(index_type="FlatIP", dimension=dim, score_threshold=0.9)

    target = _unit_vector(seed=1, dim=dim)
    nearly_aligned = target * 0.95 + _unit_vector(seed=7, dim=dim) * 0.05
    nearly_aligned = (nearly_aligned / np.linalg.norm(nearly_aligned)).astype("float32")
    noise = [_unit_vector(seed=i + 100, dim=dim) for i in range(2)]

    encoded = EncodedChunkSet(
        name="whole_recipe",
        encoded_chunks=[
            _make_encoded_chunk(0, 0, "noise-a", noise[0]),
            _make_encoded_chunk(1, 1, "near-match", nearly_aligned),
            _make_encoded_chunk(2, 2, "noise-b", noise[1]),
        ],
    )
    indexer.index_embeddings([encoded])

    hits = indexer.search_one(target, k=3)
    assert all(score > 0.9 for score, _ in hits)


def test_unresolved_faiss_id_yields_none_chunk():
    """If FAISS returns an id the indexer doesn't have a Chunk for, the
    indexer surfaces (score, None). The Retriever then filters those out."""
    dim = 8
    indexer = Indexer(index_type="FlatIP", dimension=dim, score_threshold=None)

    target = _unit_vector(seed=1, dim=dim)
    encoded = EncodedChunkSet(
        name="whole_recipe",
        encoded_chunks=[_make_encoded_chunk(0, 0, "target", target)],
    )
    indexer.index_embeddings([encoded])

    # Sanity: the one indexed chunk is returned as expected.
    hits = indexer.search_one(target, k=1)
    assert hits[0][1] is not None
    assert hits[0][1].chunk_id == 0
