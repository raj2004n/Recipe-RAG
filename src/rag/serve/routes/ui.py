"""The keyless UI routes: three forwarders into the keyed API.

The browser page never carries a key. These three routes are the whole
keyless surface apart from the page and its assets, and each forwards to
the keyed route it names (`/query`, `/health`, `/info`) with the UI's key
injected server-side, so the auth boundary is crossed in exactly one
place and the page cannot reach anything else.

The forward goes through `httpx.ASGITransport` against this same app
rather than calling the Pipeline directly. That keeps `/query` the single
entrance to the generator: the UI inherits its body validation, its
server-controlled `top_k`, and its error envelope, and the browser sees
the same status codes and `error.code` values any other client sees.

Three named routes rather than one path forwarder: a relay would serve
whatever path the caller asked for, and the explicit allowlist is what
makes this surface readable at a glance.

`/ui/query` is the one route that adds anything to the reply: it parses the
answer into the sections the result view draws and returns them alongside
the answer it was parsed from.
"""

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..parser import parse_answer

router = APIRouter()


async def _forward(request: Request, method: str, path: str) -> httpx.Response:
    """Send the browser's request onward and return the upstream reply.

    The body passes through untouched rather than being re-validated here,
    so whatever `/query` decides about it is what the browser sees.
    """
    client: httpx.AsyncClient = request.app.state.ui_client
    headers = {"X-API-Key": request.app.state.ui_key}

    content = await request.body()
    if content:
        headers["Content-Type"] = request.headers.get(
            "content-type", "application/json"
        )

    return await client.request(
        method, path, content=content or None, headers=headers
    )


def _verbatim(upstream: httpx.Response) -> JSONResponse:
    """The upstream reply, status and body, with nothing added."""
    return JSONResponse(
        status_code=upstream.status_code, content=upstream.json()
    )


@router.post("/ui/query")
async def ui_query(request: Request) -> JSONResponse:
    """The answer, plus the sections the page draws it as.

    The parse happens here rather than in the browser so that it is covered
    by the same test run as everything else. The raw answer stays in the
    payload: the page falls back to it when the parse finds no sections, and
    a bad parse is readable in the page's own network tab.
    """
    upstream = await _forward(request, "POST", "/query")
    if upstream.status_code != 200:
        return _verbatim(upstream)

    payload = upstream.json()
    payload["recipe"] = parse_answer(payload["answer"]).as_dict()
    return JSONResponse(status_code=upstream.status_code, content=payload)


@router.get("/ui/health")
async def ui_health(request: Request) -> JSONResponse:
    return _verbatim(await _forward(request, "GET", "/health"))


@router.get("/ui/info")
async def ui_info(request: Request) -> JSONResponse:
    return _verbatim(await _forward(request, "GET", "/info"))
