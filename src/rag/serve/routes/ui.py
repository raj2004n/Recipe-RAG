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
