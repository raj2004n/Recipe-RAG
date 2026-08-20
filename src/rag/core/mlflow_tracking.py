"""MLflow conventions."""

from .config import BASE_CONFIG


def build_run_params() -> dict[str, object]:
    params: dict[str, object] = {
        "rag.strategy":       BASE_CONFIG["rag"]["strategy"],
        "encoder.model_id":   BASE_CONFIG["encoder"]["model_id"],
        "encoder.batch_size": BASE_CONFIG["encoder"]["batch_size"],
        "indexer.type":       BASE_CONFIG["indexer"]["type"],
        "indexer.dimension":  BASE_CONFIG["indexer"]["dimension"],
        "retriever.top_k":    BASE_CONFIG["retriever"]["top_k"],
    }
    if "hybrid" in BASE_CONFIG:
        params["hybrid.fusion.strategy"] = BASE_CONFIG["hybrid"]["fusion"]["strategy"]
    return params
