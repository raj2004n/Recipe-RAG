async def test_auth_rejects_missing_key(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_auth_rejects_unknown_key(client):
    async with client as c:
        r = await c.get("/health", headers={"X-API-Key": "not-a-real-key"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_auth_forbidden_on_role_mismatch(client):
    """`public` must not reach a hypothetical admin-only endpoint.

    The /query endpoint accepts public, so the positive case here is
    `200 with public key`. The forbidden case is exercised directly
    against `require_role("admin")` because no current route gates on
    admin only - adding such a route later only needs this dependency.
    """
    async with client as c:
        r = await c.post(
            "/query",
            json={"query": "hi"},
            headers={"X-API-Key": "demo-public-2026"},
        )
    assert r.status_code == 200

    # Direct dependency check: a public key cannot pass require_role("admin").
    from src.rag.serve.auth import require_role
    from src.rag.serve.errors import ForbiddenError
    import pytest

    admin_only = require_role("admin")
    with pytest.raises(ForbiddenError):
        await admin_only("demo-public-2026")
