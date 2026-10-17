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
            "generator_adapter": cfg.get("generator", {}).get("adapter_path"),
        },
        "chunk_sets": {
            "dense": [cs.name for cs in snapshot.indexed_chunk_sets],
            "bm25": [cs.name for cs in snapshot.bm25_indexed_chunk_sets],
        },
    }
