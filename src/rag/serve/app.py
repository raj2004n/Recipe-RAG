"""FastAPI app + lifespan for the Recipe-RAG service.

The lifespan does the following, in order:

1. Parse `RECIPE_RAG_API_KEYS`. If missing or malformed, exit 2.
2. Load the latest sealed snapshot via `load_snapshot_by_target("latest")`.
   If no snapshot is complete, exit 2.
3. Resolve the snapshot's `dataset_key` through `LocalDatasetResolver`.
   If the file is gone or the checksum has drifted, exit 2 — there is no
   degraded mode for hybrid. (Dense also fails closed for consistency;
   the dataset is part of the snapshot's identity either way.)
4. Construct the Pipeline. Hybrid takes the resolved `recipes_path`;
   dense does not need it.
5. Warm the cross-encoder reranker for hybrid (the cold-start tax users
   actually feel). The generator stays lazy — its forward pass cannot
   share VRAM with an already-resident cross-encoder on small GPUs.
6. Store the warm Pipeline on `app.state.pipeline`.

On shutdown, call `getattr(pipeline, "close", lambda: None)()` so the
Pipeline has a future-friendly shutdown hook without forcing one today.

The whole pipeline is `def` (sync) — FastAPI runs sync handlers in a
threadpool so they do not block the event loop. With `--workers 1`, each
pod runs a single worker process.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from . import auth as auth_module
from .errors import NoSnapshotError, install_error_handlers
from .routes import health as health_route
from .routes import info as info_route
from .routes import query as query_route

logger = logging.getLogger(__name__)


def _exit_2_on_failure(message: str, exc: Exception) -> None:
    """Log + raise SystemExit(2). Used when the lifespan cannot reach a
    state where the service is safe to start."""
    logger.error("%s: %s", message, exc)
    raise SystemExit(2)


def _load_snapshot_or_exit():
    """Resolve the latest sealed snapshot, exit 2 if there isn't one."""
    from ..core.snapshot import loader

    try:
        return loader.load_snapshot_by_target("latest")
    except FileNotFoundError as exc:
        _exit_2_on_failure("No sealed snapshot to load", exc)
    except RuntimeError as exc:
        # load_snapshot_by_target raises RuntimeError when the registry
        # status is not "complete".
        _exit_2_on_failure("Latest snapshot is not complete", exc)
    except Exception as exc:
        _exit_2_on_failure("Failed to load latest snapshot", exc)


def _resolve_recipes_path_or_exit(snapshot) -> Path | None:
    """For hybrid snapshots, resolve the dataset key to a local path.

    Returns None for dense snapshots (no recipes_path required). Exits 2
    on any failure: missing key, missing file, or checksum mismatch.
    """
    if snapshot.retrieval_strategy == "dense":
        return None

    from ..core.datasets.resolver import LocalDatasetResolver
    from ..core.snapshot.paths import SnapshotPaths
    from ..core.snapshot import registry as registry_module

    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry_module.get_dataset,
    )
    try:
        resolved = resolver.resolve(snapshot.dataset_key)
    except Exception as exc:
        _exit_2_on_failure(
            f"Failed to resolve dataset {snapshot.dataset_key!r}",
            exc,
        )
    return Path(resolved.path)


def _build_pipeline(snapshot, recipes_path: Path | None) -> Any:
    """Construct the right Pipeline for the snapshot's strategy.

    Hooked so tests can monkeypatch `_build_pipeline` without spinning up
    real encoders. Production passes through to the concrete builders.
    """
    if snapshot.retrieval_strategy == "dense":
        from ..pipeline.dense_pipeline import DensePipeline
        return DensePipeline(snapshot)
    if snapshot.retrieval_strategy == "hybrid":
        from ..pipeline.hybrid_pipeline import HybridPipeline
        return HybridPipeline(snapshot, recipes_path=str(recipes_path))
    raise ValueError(
        f"Unknown retrieval strategy: {snapshot.retrieval_strategy!r}"
    )


def _eager_load_models(pipeline: Any) -> None:
    """Warm the cross-encoder reranker so the first user request does not
    pay the cold-start tax. The generator stays lazy — its forward pass
    needs VRAM headroom that some GPU classes (notably 6 GiB) cannot
    satisfy with the cross-encoder already resident. Failures here are
    fatal: the service refuses to start without a warm reranker, since
    hybrid's main quality lift comes from reranking."""
    reranker = getattr(pipeline, "reranker", None)
    if reranker is None:
        return
    try:
        reranker._ensure_model()
    except Exception as exc:
        _exit_2_on_failure("Failed to warm cross-encoder reranker", exc)


def create_app() -> FastAPI:
    """Build the FastAPI app with lifespan-warm Pipeline and the route tables."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("lifespan: starting Recipe-RAG service")

        # 1. API keys — fail closed at startup if missing or malformed.
        try:
            auth_module._load_api_keys()
        except Exception as exc:
            _exit_2_on_failure("RECIPE_RAG_API_KEYS not configured", exc)

        # 2. Load latest sealed snapshot.
        snapshot = _load_snapshot_or_exit()
        app.state.snapshot = snapshot
        logger.info(
            "lifespan: loaded snapshot strategy=%s dataset_key=%s",
            snapshot.retrieval_strategy,
            snapshot.dataset_key,
        )

        # 3. Resolve recipes path for hybrid (None for dense).
        recipes_path = _resolve_recipes_path_or_exit(snapshot)
        if recipes_path is not None:
            logger.info("lifespan: resolved recipes path to %s", recipes_path)

        # 4. Construct + warm the Pipeline.
        pipeline = _build_pipeline(snapshot, recipes_path)
        _eager_load_models(pipeline)
        app.state.pipeline = pipeline
        logger.info("lifespan: pipeline ready")

        try:
            yield
        finally:
            logger.info("lifespan: shutting down")
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="Recipe-RAG",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    install_error_handlers(app)
    app.include_router(health_route.router)
    app.include_router(info_route.router)
    app.include_router(query_route.router)
    return app


# Exposed for tests that want a real app instance without going through
# create_app (e.g. lifespan-resolves-dataset tests that drive lifespan
# manually).
__all__ = ["create_app", "_build_pipeline", "_resolve_recipes_path_or_exit"]
