import asyncio
import json

import pytest

from src.rag.serve.auth import _resolve_ui_key

from .conftest import TEST_KEYS


async def test_index_is_served_without_a_key(client):
    async with client as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text


async def test_static_assets_are_served_without_a_key(client):
    async with client as c:
        r = await c.get("/static/index.html")
    assert r.status_code == 200
    assert "<html" in r.text


async def test_every_asset_the_page_references_is_served(client):
    """The page styles and scripts itself from its own origin with no CDN and
    no fallback, so a reference that does not resolve is an unstyled page."""
    import re

    from src.rag.serve.app import STATIC_DIR

    page = (STATIC_DIR / "index.html").read_text()
    refs = re.findall(r'(?:href|src)="(/static/[^"]+)"', page)
    assert refs, "the page references no assets at all"

    async with client as c:
        for ref in refs:
            r = await c.get(ref)
            assert r.status_code == 200, ref


def test_the_page_takes_no_class_name_daisyui_already_defines():
    """A hand-written class that happens to be a DaisyUI component silently
    restyles the element: DaisyUI's .hero, for one, is a grid that stacks its
    direct children in a single cell, which laid the whole entry screen out on
    top of itself. Only the components the page means to use are allowed."""
    import re

    from src.rag.serve.app import STATIC_DIR

    intended = {
        "btn", "btn-primary", "btn-lg", "btn-sm", "btn-ghost", "btn-outline",
        "badge", "badge-sm", "textarea", "kbd", "kbd-sm",
    }

    daisyui = (STATIC_DIR / "vendor" / "daisyui.css").read_text()
    comments = re.compile(r"/\*.*?\*/", re.S)
    defined = set()
    for selector in re.findall(r"(?:^|[,{}])\s*([^{}@]*)\{", comments.sub(" ", daisyui)):
        defined.update(re.findall(r"\.([A-Za-z][A-Za-z0-9_-]*)", selector))
    assert len(defined) > 100, "the vendored stylesheet did not parse"

    page = (STATIC_DIR / "index.html").read_text()
    used = set()
    for value in re.findall(r'class="([^"]*)"', page):
        used.update(value.split())
    # Comments are stripped first, or a class named in a comment would read as
    # a definition.
    own_css = set(
        re.findall(
            r"\.([A-Za-z][A-Za-z0-9_-]*)",
            comments.sub(" ", (STATIC_DIR / "app.css").read_text()),
        )
    )

    assert sorted((used | own_css) & defined - intended) == []


async def test_ui_query_answers_without_a_key(client):
    """The browser holds no key; the forward injects one. The recorded call
    carries /query's own server-controlled k, which is what shows the
    request went through the route and not straight at the Pipeline."""
    async with client as c:
        r = await c.post(
            "/ui/query", json={"query": "what can I make with chicken?"}
        )
        pipeline = c._transport.app.state.pipeline  # noqa: SLF001 - test-only

    assert r.status_code == 200
    payload = r.json()
    assert payload["query"] == "what can I make with chicken?"
    assert payload["answer"] == "stub answer for: what can I make with chicken?"
    assert payload["recipe"] == {
        "title": "",
        "meta": [],
        "ingredients": [],
        "steps": [],
        "notes": [],
        "remainder": payload["answer"],
        "parsed": False,
    }
    assert pipeline.query_calls == [
        {"query": "what can I make with chicken?", "k": 3}
    ]


async def test_ui_query_returns_the_sections_it_parsed(client):
    """The page draws a recipe rather than a wall of text, so the forward
    parses the answer for it. The answer it was parsed from stays in the
    payload, which is what makes the page's fallback to prose one branch
    rather than a second request."""
    answer = (
        "## Lemon Chicken\n"
        "\n"
        "**Yields:** 4 servings\n"
        "\n"
        "**Ingredients:**\n"
        "*   1 lemon\n"
        "\n"
        "**Instructions:**\n"
        "1. Cook it.\n"
    )
    async with client as c:
        c._transport.app.state.pipeline.query = (  # noqa: SLF001
            lambda query, k=3: {"query": query, "answer": answer}
        )
        r = await c.post("/ui/query", json={"query": "anything"})

    assert r.status_code == 200
    payload = r.json()
    assert payload["answer"] == answer
    assert payload["recipe"]["parsed"] is True
    assert payload["recipe"]["title"] == "Lemon Chicken"
    assert payload["recipe"]["meta"] == [
        {"label": "Yields", "value": "4 servings"}
    ]
    assert payload["recipe"]["ingredients"] == [
        {"kind": "item", "quantity": "1", "text": "lemon"}
    ]
    assert payload["recipe"]["steps"] == [
        {"kind": "step", "text": "Cook it."}
    ]


async def test_ui_query_validation_comes_from_the_upstream_route(client):
    """An empty query is rejected by /query, not by a rule copied into the
    forwarder. The envelope is the one /query's validation handler builds,
    and nothing is added to a reply that has no answer in it."""
    async with client as c:
        r = await c.post("/ui/query", json={"query": ""})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"
    assert "recipe" not in r.json()


async def test_ui_query_authenticates_with_the_ui_key(client):
    """Break the UI key and the upstream call comes back 401. That is what
    shows the forward authenticates with `app.state.ui_key`, not with
    anything the browser sent."""
    async with client as c:
        c._transport.app.state.ui_key = "not-a-real-key"  # noqa: SLF001
        r = await c.post("/ui/query", json={"query": "anything"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthenticated"


async def test_ui_health_and_info_match_the_keyed_routes(client, auth_header):
    async with client as c:
        ui_health = await c.get("/ui/health")
        keyed_health = await c.get("/health", headers=auth_header())
        ui_info = await c.get("/ui/info")
        keyed_info = await c.get("/info", headers=auth_header())

    assert ui_health.status_code == keyed_health.status_code == 200
    assert ui_health.json() == keyed_health.json()
    assert ui_info.json() == keyed_info.json()


async def test_ui_health_passes_an_upstream_503_through(
    monkeypatch, stub_snapshot
):
    """A service with no pipeline reports 503 no_snapshot to the browser,
    not a 500 from the forward itself."""
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
            r = await c.get("/ui/health")

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "no_snapshot"


async def test_forwarder_client_is_closed_on_shutdown(stub_pipeline, monkeypatch):
    """The forwarder's client belongs to the lifespan, so shutdown closes
    it along with the Pipeline."""
    from src.rag.serve import app as app_module

    monkeypatch.setenv("RECIPE_RAG_API_KEYS", json.dumps(TEST_KEYS))
    app = app_module.create_app()

    async with app.router.lifespan_context(app):
        forwarder = app.state.ui_client
        assert not forwarder.is_closed

    assert forwarder.is_closed


def test_lifespan_exits_2_when_the_ui_key_is_not_in_the_map(
    stub_pipeline, monkeypatch
):
    """RAMSAY_UI_KEY naming a key that is not configured stops the service
    before it serves, the way a malformed key map does."""
    from src.rag.serve import app as app_module

    monkeypatch.setenv("RECIPE_RAG_API_KEYS", json.dumps(TEST_KEYS))
    monkeypatch.setenv("RAMSAY_UI_KEY", "a-key-that-is-not-configured")
    app = app_module.create_app()

    async def _enter():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(_enter())
    assert exc_info.value.code == 2


def test_resolve_ui_key_prefers_the_named_key(monkeypatch):
    monkeypatch.setenv("RAMSAY_UI_KEY", "ops-admin-2026")
    assert _resolve_ui_key(TEST_KEYS) == "ops-admin-2026"


def test_resolve_ui_key_falls_back_to_the_first_public_key(monkeypatch):
    monkeypatch.delenv("RAMSAY_UI_KEY", raising=False)
    assert _resolve_ui_key(TEST_KEYS) == "demo-public-2026"


def test_resolve_ui_key_rejects_a_name_that_is_not_configured(monkeypatch):
    monkeypatch.setenv("RAMSAY_UI_KEY", "not-a-configured-key")
    with pytest.raises(RuntimeError):
        _resolve_ui_key(TEST_KEYS)


def test_resolve_ui_key_needs_a_public_key_when_unnamed(monkeypatch):
    monkeypatch.delenv("RAMSAY_UI_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _resolve_ui_key({"ops-admin-2026": "admin"})
