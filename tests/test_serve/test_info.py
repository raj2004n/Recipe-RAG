"""Tests for GET /info."""


async def test_info_returns_snapshot_metadata(client, auth_header):
    async with client as c:
        r = await c.get("/info", headers=auth_header())
    assert r.status_code == 200
    body = r.json()
    assert body["snapshot"]["retrieval_strategy"] == "dense"
    assert body["snapshot"]["dataset_key"] == "test-dataset"
    assert body["models"]["encoder"] == "sentence-transformers/test"
    assert body["models"]["generator"] == "test/generator"
    assert body["models"]["generator_adapter"] == "test/adapter"
    assert body["chunk_sets"]["dense"] == []
    assert body["chunk_sets"]["bm25"] == []
