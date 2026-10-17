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
