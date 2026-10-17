from types import SimpleNamespace
from unittest.mock import patch

from src.runtime.hybrid_pipeline import HybridPipeline
from src.domain.types import Chunk, FusedChunk


def _fake_snapshot():
    return SimpleNamespace(
        retrieval_strategy="hybrid",
        base_config={
            "encoder": {"model_id": "enc", "batch_size": 1, "normalize_embeddings": True},
            "indexer": {"type": "FlatIP", "dimension": 4},
            "retriever": {"score_threshold": 0.0},
            "hybrid": {
                "retrieval": {"top_k_dense": 5, "top_k_sparse": 5},
                "fusion": {"strategy": "rrf", "rrf_k": 60, "weight_dense": 1.0, "weight_sparse": 1.0},
                "reranker": {"model_id": "rerank", "max_length": 64, "top_n": 10},
                "prompt": {"final_top_k": 3},
            },
            "generator": {
                "model_id": "gen", "max_input_tokens": 100, "max_new_tokens": 10,
                "truncation": True, "return_tensors": "pt", "adapter_path": None,
            },
        },
        indexed_chunk_sets=[],
        bm25_indexed_chunk_sets=[object()],
        prompts_config={"settings": {"max_context_tokens": 100}, "template": {}},
    )


def _chunk(chunk_id: int, recipe_id: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        recipe_id=recipe_id,
        chunk_type="ingredients",
        text="x",
    )


def _fused(chunk: Chunk) -> FusedChunk:
    return FusedChunk(chunk=chunk, fused_score=1.0, sources=("dense",))


@patch("src.runtime.hybrid_pipeline.AutoTokenizer")
@patch("src.runtime.hybrid_pipeline.PromptBuilder")
@patch("src.runtime.hybrid_pipeline.CrossEncoderReranker")
@patch("src.runtime.hybrid_pipeline.BM25Retriever")
@patch("src.runtime.hybrid_pipeline.Indexer")
@patch("src.runtime.hybrid_pipeline.Encoder")
def test_debug_query_returns_deduped_recipes_in_rank_order(*_mocks):
    recipes = [
        {"recipe_id": 1, "recipe_name": "one"},
        {"recipe_id": 2, "recipe_name": "two"},
        {"recipe_id": 3, "recipe_name": "three"},
    ]
    pipeline = HybridPipeline(_fake_snapshot(), recipes=recipes)
    pipeline.dense_retriever.retrieve = lambda query: []
    pipeline.bm25_retriever.retrieve = lambda query: []

    # The reranker returns chunks with a duplicate recipe_id (recipe 1 has
    # both an ingredients and an info chunk); _fetch_recipes must dedupe.
    pipeline.reranker.rerank.return_value = [
        _fused(_chunk(10, 1)),
        _fused(_chunk(11, 2)),
        _fused(_chunk(12, 1)),  # duplicate recipe_id
        _fused(_chunk(13, 3)),
    ]
    pipeline.prompt_builder.build_prompt_from_recipes.return_value = ("system", "user")

    recipes_out, _, prompts = pipeline.debug_query("query", k=3)

    assert [r["recipe_id"] for r in recipes_out] == [1, 2, 3]
    assert prompts == ("system", "user")
    # debug_query must not touch the generator.
    assert pipeline.generator is None


@patch("src.runtime.hybrid_pipeline.AutoTokenizer")
@patch("src.runtime.hybrid_pipeline.PromptBuilder")
@patch("src.runtime.hybrid_pipeline.CrossEncoderReranker")
@patch("src.runtime.hybrid_pipeline.BM25Retriever")
@patch("src.runtime.hybrid_pipeline.Indexer")
@patch("src.runtime.hybrid_pipeline.Encoder")
def test_debug_query_respects_k(*_mocks):
    recipes = [
        {"recipe_id": 1, "recipe_name": "one"},
        {"recipe_id": 2, "recipe_name": "two"},
        {"recipe_id": 3, "recipe_name": "three"},
    ]
    pipeline = HybridPipeline(_fake_snapshot(), recipes=recipes)
    pipeline.dense_retriever.retrieve = lambda query: []
    pipeline.bm25_retriever.retrieve = lambda query: []
    pipeline.reranker.rerank.return_value = [
        _fused(_chunk(10, 1)),
        _fused(_chunk(11, 2)),
        _fused(_chunk(12, 3)),
    ]
    pipeline.prompt_builder.build_prompt_from_recipes.return_value = ("system", "user")

    recipes_out, _, _ = pipeline.debug_query("query", k=2)

    assert [r["recipe_id"] for r in recipes_out] == [1, 2]
