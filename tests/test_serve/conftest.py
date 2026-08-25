"""Fixtures and helpers for the FastAPI service tests."""

import json
import os
from typing import Any

import httpx
import pytest


TEST_KEYS = {
    "demo-public-2026": "public",
    "ops-admin-2026": "admin",
}


# Set the env var at module import so tests that don't use the `client`
# fixture (e.g. test_auth.py asserting on auth directly) still have the
# keys configured. monkeypatch overrides this within the fixture scope.
os.environ.setdefault("RECIPE_RAG_API_KEYS", json.dumps(TEST_KEYS))


class StubSnapshot:
    """Stand-in for the snapshot loader's return value.

    Tests pass it directly into `_build_pipeline` so we do not exercise the
    snapshot subsystem in serve tests. The fields mirror the real Snapshot
    dataclass's surface that the routes actually read.
    """

    def __init__(
        self,
        *,
        retrieval_strategy: str = "dense",
        dataset_key: str = "test-dataset",
        created_at: str = "2026-08-23 11:42:01",
    ):
        self.retrieval_strategy = retrieval_strategy
        self.dataset_key = dataset_key
        self.created_at = created_at
        self.base_config = {
            "encoder": {"model_id": "sentence-transformers/test"},
            "generator": {"model_id": "test/generator"},
            "indexer": {"type": "flatip", "dimension": 4},
            "retriever": {"top_k": 3, "score_threshold": 0.0},
            "hybrid": {
                "retrieval": {"top_k_dense": 3},
                "fusion": {"strategy": "rrf", "rrf_k": 60,
                           "weight_dense": 0.5, "weight_sparse": 0.5},
                "reranker": {"model_id": "test/reranker",
                             "max_length": 256, "top_n": 3},
                "prompt": {"final_top_k": 3},
            },
        }
        self.prompts_config = {"settings": {"max_context_tokens": 1024}, "template": {}}
        self.indexed_chunk_sets: list = []
        self.bm25_indexed_chunk_sets: list = []


class StubPipeline:
    """Stand-in for DensePipeline / HybridPipeline.

    Stores query calls so tests can assert on them. `load_generator` and
    `close` are no-ops so the lifespan completes without error.
    """

    def __init__(self, snapshot: Any, **kwargs):
        self._snapshot = snapshot
        self.query_calls: list[dict] = []
        self.closed = False

    def load_generator(self):
        return None

    def close(self):
        self.closed = True

    def query(self, query: str, k: int = 3):
        self.query_calls.append({"query": query, "k": k})
        return {"query": query, "answer": f"stub answer for: {query}"}


@pytest.fixture
def stub_snapshot():
    return StubSnapshot()


@pytest.fixture
def stub_pipeline(stub_snapshot, monkeypatch):
    """Patch `_build_pipeline` to return a StubPipeline so the lifespan
    does not pull in real encoders. Also patches the lifespan's
    `_load_snapshot_or_exit` and `_resolve_recipes_path_or_exit` so we
    do not touch the real registry / resolver.
    """
    from src.rag.serve import app as app_module

    def fake_build_pipeline(snapshot, recipes_path=None):
        return StubPipeline(snapshot)

    monkeypatch.setattr(app_module, "_build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(app_module, "_load_snapshot_or_exit", lambda: stub_snapshot)
    monkeypatch.setattr(app_module, "_resolve_recipes_path_or_exit", lambda s: None)
    return StubPipeline


@pytest.fixture
async def client(stub_pipeline, monkeypatch):
    """An httpx AsyncClient bound to a freshly-built app with the lifespan
    driven manually. `httpx.ASGITransport` does not run ASGI lifespan by
    default, so we enter `app.router.lifespan_context(app)` ourselves.

    The resulting `app.state.pipeline` is the StubPipeline patched above.
    RECIPE_RAG_API_KEYS is set so the lifespan's auth check passes.
    """
    monkeypatch.setenv("RECIPE_RAG_API_KEYS", repr(TEST_KEYS).replace("'", '"'))
    from src.rag.serve.app import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        yield httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def auth_header():
    """Return an `X-API-Key` header for the public role by default."""

    def _make(role: str = "public") -> dict[str, str]:
        key = next(k for k, v in TEST_KEYS.items() if v == role)
        return {"X-API-Key": key}

    return _make
