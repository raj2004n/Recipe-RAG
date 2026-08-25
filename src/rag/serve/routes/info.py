"""GET /info - read-only snapshot metadata for operators.

Returns the same data /info consumers would want when triaging a deploy:
which snapshot is loaded, which dataset it was built against, which
model ids it carries, and the chunk-set names. Deliberately excludes
retrieval hits and prompts - those are /query's concern, not /info's.
"""

from fastapi import APIRouter, Depends, Request

from ..auth import require_role

router = APIRouter()


@router.get("/info", dependencies=[Depends(require_role("public"))])
def info(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    snapshot = pipeline._snapshot
    cfg = snapshot.base_config
    return {
        "snapshot": {
            "retrieval_strategy": snapshot.retrieval_strategy,
            "created_at": snapshot.created_at,
            "dataset_key": snapshot.dataset_key,
        },
        "models": {
            "encoder": cfg.get("encoder", {}).get("model_id"),
            "generator": cfg.get("generator", {}).get("model_id"),
        },
        "chunk_sets": {
            "dense": [cs.name for cs in snapshot.indexed_chunk_sets],
            "bm25": [cs.name for cs in snapshot.bm25_indexed_chunk_sets],
        },
    }
