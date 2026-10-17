async def test_query_returns_answer(client, auth_header):
    async with client as c:
        r = await c.post(
            "/query",
            json={"query": "what can I make with chicken?"},
            headers=auth_header(),
        )
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "what can I make with chicken?"
    assert body["answer"].startswith("stub answer for: ")


async def test_query_rejects_empty_query(client, auth_header):
    async with client as c:
        r = await c.post(
            "/query",
            json={"query": ""},
            headers=auth_header(),
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


async def test_query_internal_error_scrubs_traceback(client, auth_header, monkeypatch):
    """An unhandled exception in the route must not leak the exception
    message or any internal identifier in the response body - only the
    generic envelope. The full detail goes to the server log, not the wire."""
    async with client as c:
        pipeline = c._transport.app.state.pipeline  # noqa: SLF001 - test-only
        pipeline.query = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("CUDA OOM fingerprint=0xdeadbeef gpu_secret=42")
        )
        r = await c.post(
            "/query",
            json={"query": "anything"},
            headers=auth_header(),
        )

    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # The body is the generic envelope - no internal strings, identifiers,
    # or exception message must leak through.
    message = body["error"]["message"]
    assert "CUDA" not in message
    assert "0xdeadbeef" not in message
    assert "gpu_secret" not in message


def _explode_factory():
    """Helper - unused; lint pacifier for the monkeypatch arg shape above."""
    raise NotImplementedError
