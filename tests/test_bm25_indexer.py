"""Tests for the BM25 indexer.

The BM25 indexer turns a TokenizedChunkSet into a BM25IndexedChunkSet whose
index is a rank_bm25.BM25Okapi and whose `chunks` map is the chunk_id -> Chunk
resolution table the retriever needs to turn a hit into a RetrievalResult.
These tests pin both contracts: the ranking behaviour and the chunk map.
"""

from src.rag.retrieval.chunker import Chunk
from src.rag.retrieval.sparse.bm25indexer import BM25Indexer
from src.rag.retrieval.sparse.tokenizer import Tokenizer, TokenizedChunk, TokenizedChunkSet


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        recipe_id=chunk_id,
        chunk_type="ingredient",
        text=text,
    )


def _tokenized(chunk_id: int, text: str) -> TokenizedChunk:
    return TokenizedChunk(chunk=_chunk(chunk_id, text), tokens=Tokenizer().tokenize(text))


def test_bm25_search_finds_exact_term_match():
    """Indexing chunks with disjoint vocabulary and querying with one chunk's
    exact terms should put that chunk strictly above the others.

    Note on corpus size: BM25's IDF for a term is log((N - df + 0.5) / (df + 0.5)),
    which only goes positive when N > 2*df. With N=2 and df=1 the IDF is exactly
    log(1)=0 and the test would degenerate to score[0]==score[1]==0.0 regardless
    of which chunk matches. We use N=3 with one matching chunk and two
    non-matching noise chunks so the matching chunk has df=1 against N=3, giving
    a positive IDF and a strict ranking. The production corpus is hundreds of
    recipes, well clear of this small-N regime.
    """
    indexer = BM25Indexer()
    chunk_set = TokenizedChunkSet(
        name="ingredients",
        tokenized_chunks=[
            _tokenized(0, "chicken tikka masala"),
            _tokenized(1, "beef bourguignon with mushrooms"),
            _tokenized(2, "lentil dal with turmeric"),
        ],
    )

    [indexed] = indexer.build(chunk_set)
    query_tokens = Tokenizer().tokenize("tikka masala")
    scores = indexed.index.get_scores(query_tokens)

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_build_populates_chunks_by_id():
    """After build(), the returned BM25IndexedChunkSet's `chunks` map is
    keyed by chunk_id and points at the same Chunk objects the input
    TokenizedChunks carried. The retriever relies on this map to resolve
    a hit index back to a Chunk."""
    indexer = BM25Indexer()
    chunk_a = _tokenized(7, "garlic and ginger paste")
    chunk_b = _tokenized(42, "fish sauce and lime")
    chunk_set = TokenizedChunkSet(
        name="ingredients",
        tokenized_chunks=[chunk_a, chunk_b],
    )

    [indexed] = indexer.build(chunk_set)

    assert set(indexed.chunks.keys()) == {7, 42}
    assert indexed.chunks[7] is chunk_a.chunk
    assert indexed.chunks[42] is chunk_b.chunk


def test_bm25_search_returns_zero_scores_on_no_match():
    """A query with no token overlap against the indexed vocabulary yields
    zero scores everywhere - the retriever filters those out so the call
    site sees an empty list."""
    indexer = BM25Indexer()
    chunk_set = TokenizedChunkSet(
        name="ingredients",
        tokenized_chunks=[
            _tokenized(0, "chicken tikka masala"),
            _tokenized(1, "beef bourguignon"),
        ],
    )

    [indexed] = indexer.build(chunk_set)
    unrelated_tokens = Tokenizer().tokenize("quantum chromodynamics")
    scores = indexed.index.get_scores(unrelated_tokens)

    assert all(score == 0 for score in scores)
