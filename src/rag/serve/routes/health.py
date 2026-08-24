"""GET /health — readiness probe for the deploy workflow.

Returns 200 with `{"status": "ok", ...}` when a Pipeline is loaded onto
`app.state.pipeline`. Returns 503 with `{"status": "no_snapshot", ...}`
otherwise — the deploy smoke-test treats 5xx as a deploy failure.

This endpoint does not touch the Pipeline. A loaded Pipeline does not
mean a query will succeed (a generator hang is a different failure mode,
flagged in the plan's known risks). What it tells the caller is: the
snapshot subsystem is healthy and the service has a model to query.
"""

from fastapi import APIRouter, Depends, Request

from ..auth import require_role
from ..errors import NoSnapshotError

router = APIRouter()


@router.get("/health", dependencies=[Depends(require_role("public"))])
def health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise NoSnapshotError(
            "No pipeline is loaded. Refusing to report healthy."
        )
    snapshot = pipeline._snapshot
    return {
        "status": "ok",
        "snapshot_loaded": True,
        "snapshot_name": snapshot.retrieval_strategy,
        "created_at": snapshot.created_at,
    }
