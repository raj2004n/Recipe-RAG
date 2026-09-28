"""Tests for the lifespan's fail-closed posture on missing dataset (Risk 2).

The lifespan must exit 2 when the dataset registered for the snapshot is
no longer at its registered path (file deleted, volume not mounted, cloud
object versioned out). This is what stops the service from running in a
silently-degraded state where /health is green but /query cannot answer.
"""

import pytest

from src.rag.core.snapshot import registry as registry_module
from src.rag.core.snapshot.paths import SnapshotPaths


def _seed_registry_with_dataset(tmp_path, *, dataset_path, checksum):
    """Write a registry file that points at dataset_path with the given
    checksum. The resolver will recompute the file's checksum and detect
    drift when what is on disk does not match."""
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    registry_module.upsert_dataset(
        registry_file,
        "test-dataset",
        registry_module.DatasetEntry(
            path=str(dataset_path),
            created_at="2026-08-23 11:42:01",
            checksum=checksum,
            size_bytes=0,
        ),
    )
    return registry_file


def test_lifespan_exits_2_when_dataset_missing(tmp_path, monkeypatch):
    """Drive the lifespan manually with a snapshot whose dataset is gone.

    Expectation: SystemExit(2) is raised before the app starts serving.
    The /query endpoint would be dead anyway; the value of exiting 2 is
    that the deploy smoke-test sees a non-zero pod exit and the orchestrator
    marks the deploy as failed rather than rolling out a half-working pod.
    """
    # Build a registry that points at a path that does not exist.
    bogus_path = tmp_path / "never_existed.json"
    _seed_registry_with_dataset(
        tmp_path, dataset_path=bogus_path, checksum="sha256:deadbeef"
    )

    # Patch SnapshotPaths.from_paths_config to use our tmp registry.
    def fake_from_paths_config(paths_cfg, versioned_name=None):
        return SnapshotPaths(
            building=tmp_path / "snapshots" / "fake_build",
            chunks=tmp_path / "snapshots" / "fake_build" / "chunks",
            transformed_chunks=tmp_path / "snapshots" / "fake_build" / "transformed_chunks",
            index=tmp_path / "snapshots" / "fake_build" / "index",
            configs=tmp_path / "snapshots" / "fake_build" / "configs",
            metadata=tmp_path / "snapshots" / "fake_build" / "metadata.json",
            registry_file=tmp_path / "snapshots" / "registry.json",
            config_files=(),
        )

    monkeypatch.setattr(SnapshotPaths, "from_paths_config", fake_from_paths_config)

    # Patch load_snapshot_by_target to return a fake snapshot referencing
    # the missing dataset.
    from src.rag.core.snapshot import loader
    from src.rag.serve import app as app_module

    class _Stub:
        retrieval_strategy = "hybrid"
        dataset_key = "test-dataset"
        created_at = "2026-08-23 11:42:01"
        base_config = {"retriever": {"top_k": 3}}
        prompts_config = {"settings": {"max_context_tokens": 1024}, "template": {}}
        indexed_chunk_sets = []
        bm25_indexed_chunk_sets = []

    monkeypatch.setattr(loader, "load_snapshot_by_target", lambda target="serve": _Stub())

    monkeypatch.setenv("RECIPE_RAG_API_KEYS", '{"demo-public-2026": "public"}')
    app = app_module.create_app()

    # Drive the lifespan by entering the async context manager and
    # asserting it raises SystemExit(2).
    import asyncio

    async def _enter():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(_enter())
    assert exc_info.value.code == 2


def test_lifespan_exits_2_on_checksum_drift(tmp_path, monkeypatch):
    """The registry entry claims one checksum; the file on disk has another.
    Resolver raises ChecksumMismatchError; lifespan exits 2."""
    real_file = tmp_path / "recipes.json"
    real_file.write_text('[]')
    _seed_registry_with_dataset(
        tmp_path, dataset_path=real_file, checksum="sha256:wrong-checksum"
    )

    def fake_from_paths_config(paths_cfg, versioned_name=None):
        return SnapshotPaths(
            building=tmp_path / "snapshots" / "fake_build",
            chunks=tmp_path / "snapshots" / "fake_build" / "chunks",
            transformed_chunks=tmp_path / "snapshots" / "fake_build" / "transformed_chunks",
            index=tmp_path / "snapshots" / "fake_build" / "index",
            configs=tmp_path / "snapshots" / "fake_build" / "configs",
            metadata=tmp_path / "snapshots" / "fake_build" / "metadata.json",
            registry_file=tmp_path / "snapshots" / "registry.json",
            config_files=(),
        )

    monkeypatch.setattr(SnapshotPaths, "from_paths_config", fake_from_paths_config)

    from src.rag.core.snapshot import loader
    from src.rag.serve import app as app_module

    class _Stub:
        retrieval_strategy = "hybrid"
        dataset_key = "test-dataset"
        created_at = "2026-08-23 11:42:01"
        base_config = {"retriever": {"top_k": 3}}
        prompts_config = {"settings": {"max_context_tokens": 1024}, "template": {}}
        indexed_chunk_sets = []
        bm25_indexed_chunk_sets = []

    monkeypatch.setattr(loader, "load_snapshot_by_target", lambda target="serve": _Stub())
    monkeypatch.setenv("RECIPE_RAG_API_KEYS", '{"demo-public-2026": "public"}')
    app = app_module.create_app()

    import asyncio

    async def _enter():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(_enter())
    assert exc_info.value.code == 2


def test_lifespan_exits_2_when_api_keys_missing(tmp_path, monkeypatch):
    """The service refuses to start without RECIPE_RAG_API_KEYS. This is what
    stops the deploy workflow from rolling out an open service."""
    monkeypatch.delenv("RECIPE_RAG_API_KEYS", raising=False)

    # Use the existing stub pipeline so the snapshot subsystem is not
    # exercised - we only care that the auth check trips first.
    from src.rag.serve import app as app_module

    class _StubPipeline:
        def __init__(self, snapshot, **kw): pass
        def load_generator(self): pass

    class _StubSnapshot:
        retrieval_strategy = "dense"
        dataset_key = "test-dataset"
        created_at = "2026-08-23 11:42:01"
        base_config = {"retriever": {"top_k": 3}}
        prompts_config = {"settings": {"max_context_tokens": 1024}, "template": {}}
        indexed_chunk_sets = []
        bm25_indexed_chunk_sets = []

    monkeypatch.setattr(app_module, "_build_pipeline",
                        lambda s, r=None: _StubPipeline(s))
    monkeypatch.setattr(app_module, "_load_snapshot_or_exit",
                        lambda: _StubSnapshot())
    monkeypatch.setattr(app_module, "_resolve_recipes_path_or_exit",
                        lambda s: None)
    monkeypatch.setattr(app_module, "_eager_load_models",
                        lambda p: None)

    app = app_module.create_app()
    import asyncio

    async def _enter():
        async with app.router.lifespan_context(app):
            pass

    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(_enter())
    assert exc_info.value.code == 2


def test_lifespan_serves_the_promoted_snapshot(tmp_path, monkeypatch):
    """The lifespan answers from the `serve` target.

    That target names the folder a human promoted, so the service starts from
    what was promoted rather than from whatever was built most recently. The
    image bakes the same target, which is what keeps the snapshot the image
    carries and the snapshot the app loads from diverging.
    """
    from src.rag.core.snapshot import loader
    from src.rag.serve import app as app_module

    class _StubSnapshot:
        retrieval_strategy = "dense"
        dataset_key = "test-dataset"
        created_at = "2026-08-23 11:42:01"
        base_config = {"retriever": {"top_k": 3}}
        prompts_config = {"settings": {"max_context_tokens": 1024}, "template": {}}
        indexed_chunk_sets = []
        bm25_indexed_chunk_sets = []

    asked: list[str] = []

    def _record_target(target="latest_build"):
        asked.append(target)
        return _StubSnapshot()

    monkeypatch.setattr(loader, "load_snapshot_by_target", _record_target)
    monkeypatch.setattr(app_module, "_build_pipeline", lambda s, r=None: object())
    monkeypatch.setattr(app_module, "_eager_load_models", lambda p: None)
    monkeypatch.setenv("RECIPE_RAG_API_KEYS", '{"demo-public-2026": "public"}')

    app = app_module.create_app()

    import asyncio

    async def _enter():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(_enter())
    assert asked == ["serve"]
