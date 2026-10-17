from .config import BASE_CONFIG


def build_run_params() -> dict[str, object]:
    """The params a build run records.

    The retrieval entries come from the config blocks the index is built from.
    The generator entries name the model that will answer from that index, which
    is the part of a build's identity that no artifact in the snapshot holds, so
    the run's params are what a later audit reads to learn which generator a
    snapshot was built to serve.
    """
    params: dict[str, object] = {
        "rag.strategy":       BASE_CONFIG["rag"]["strategy"],
        "encoder.model_id":   BASE_CONFIG["encoder"]["model_id"],
        "encoder.batch_size": BASE_CONFIG["encoder"]["batch_size"],
        "indexer.type":       BASE_CONFIG["indexer"]["type"],
        "indexer.dimension":  BASE_CONFIG["indexer"]["dimension"],
        "retriever.top_k":    BASE_CONFIG["retriever"]["top_k"],
        "generator.model_id": BASE_CONFIG["generator"]["model_id"],
    }
    if "hybrid" in BASE_CONFIG:
        params["hybrid.fusion.strategy"] = BASE_CONFIG["hybrid"]["fusion"]["strategy"]
    adapter_path = BASE_CONFIG["generator"].get("adapter_path")
    if adapter_path:
        params["generator.adapter_path"] = adapter_path
    return params
