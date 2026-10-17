async def test_health_ok_when_pipeline_ready(client, auth_header):
    async with client as c:
        r = await c.get("/health", headers=auth_header())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["snapshot_loaded"] is True


async def test_health_503_when_no_snapshot(monkeypatch, stub_snapshot):
    """If the lifespan produced no pipeline (e.g. snapshot missing), the
    health endpoint reports no_snapshot at 503 so the deploy smoke-test
    catches the failure. Drive the lifespan manually with a stub that
    returns None from _build_pipeline (and bypass eager-load)."""
    import httpx
    from src.rag.serve import app as app_module

    monkeypatch.setattr(app_module, "_build_pipeline", lambda s, r=None: None)
    monkeypatch.setattr(app_module, "_load_snapshot_or_exit", lambda: stub_snapshot)
    monkeypatch.setattr(app_module, "_resolve_recipes_path_or_exit", lambda s: None)
    monkeypatch.setattr(app_module, "_eager_load_models", lambda p: None)
    app = app_module.create_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                "/health",
                headers={"X-API-Key": "demo-public-2026"},
            )

    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "no_snapshot"
