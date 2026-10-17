import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base class for service errors that map to a specific status + code."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NoSnapshotError(ServiceError):
    status_code = 503
    code = "no_snapshot"


class UnauthenticatedError(ServiceError):
    status_code = 401
    code = "unauthenticated"


class ForbiddenError(ServiceError):
    status_code = 403
    code = "forbidden"


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


class _LastResortMiddleware(BaseHTTPMiddleware):
    """Catch everything the per-route handlers don't see.

    Logs the full traceback server-side and returns the generic
    `internal_error` envelope so exception messages, traceback strings,
    and any internal identifiers never make it onto the wire.
    """

    async def dispatch(self, request: StarletteRequest, call_next):
        try:
            return await call_next(request)
        except Exception:
            logger.exception("unhandled error serving request")
            return JSONResponse(
                status_code=500,
                content=_envelope(
                    "internal_error", "An internal error occurred."
                ),
            )


def install_error_handlers(app: FastAPI) -> None:
    """Wire the exception handlers and last-resort middleware."""

    app.add_middleware(_LastResortMiddleware)

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_envelope("validation_error", str(exc)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
        # Map the common starlette codes to our codes where they line up.
        if exc.status_code == 401:
            code = "unauthenticated"
        elif exc.status_code == 403:
            code = "forbidden"
        else:
            code = "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, str(exc.detail)),
        )
