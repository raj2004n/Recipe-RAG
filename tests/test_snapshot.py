"""Tests for the snapshot lifecycle roundtrip.

Verifies that a snapshot sealed by lifecycle.finalise_snapshot can be loaded
back by loader.load_snapshot and that the metadata, configs, and chunk-set
mapping survive the trip.

The test builds the snapshot's directory structure directly under tmp_path
rather than running the full Builder, so it exercises only the seal/load
contract that the inference side depends on.
"""

import json
from pathlib import Path

import pytest

from src.rag.core.snapshot import lifecycle, loader
from src.rag.retrieval.dense.indexer import IndexedChunkSet
from src.rag.retrieval.sparse.bm25indexer import BM25IndexedChunkSet


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)


def _seed_snapshot(snapshot_dir: Path, strategy: str, dataset_key: str = "test-dataset") -> Path:
    """Lay down the minimum files a sealed snapshot needs and finalise it.

    Returns the snapshot folder path.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    metadata = snapshot_dir / "metadata.json"
    metadata.write_text(json.dumps({"strategy": strategy, "dataset_key": dataset_key}))

    # chunks/ and index/ directories are required to exist (lifecycle
    # makes the snapshot read-only on finalise). They can be empty for
    # this test because the index loaders are mocked below.
    (snapshot_dir / "chunks").mkdir()
    (snapshot_dir / "index").mkdir()

    configs_dir = snapshot_dir / "configs"
    _write_yaml(configs_dir / "base_config.yaml", {"rag": {"strategy": strategy}})
    _write_yaml(configs_dir / "prompts_config.yaml", {"personas": {}})

    return snapshot_dir


def _patch_loaders(monkeypatch, *, dense: bool, hybrid: bool) -> None:
    """Stub out the FAISS and BM25 index loaders so we don't need a real
    sealed snapshot's index payload to exercise the lifecycle.

    The loader module imports these names directly (`from .indices import
    load_bm25_index_chunk_sets, load_indexed_chunk_sets`) and dispatches
    via the bound names in its own namespace, so the patch has to land on
    `loader`, not on `indices`.
    """
    def fake_load_indexed_chunk_sets(save_path, chunks_save_path=None):
        if dense:
            return [IndexedChunkSet(name="whole_recipe", index=None, chunks={})], {
                "whole_recipe": {0: None, 1: None}
            }
        # Hybrid: a "dense" indexed chunk set.
        return [IndexedChunkSet(name="dense", index=None, chunks={})], {
            "dense": {0: None, 1: None}
        }

    def fake_load_bm25_index_chunk_sets(save_path, chunks_save_path=None):
        return (
            [BM25IndexedChunkSet(name="sparse", index=None, chunks={})],
            {"sparse": {0: None, 1: None}},
        )

    monkeypatch.setattr(
        loader, "load_indexed_chunk_sets", fake_load_indexed_chunk_sets
    )
    monkeypatch.setattr(
        loader, "load_bm25_index_chunk_sets", fake_load_bm25_index_chunk_sets
    )
    # The strategy dispatch table holds references to the original loader
    # functions captured at module import time; rebinding the module
    # attribute doesn't update the table. Patch the table entries too.
    loader._STRATEGY_LOADERS["dense"] = [fake_load_indexed_chunk_sets]
    if hybrid:
        loader._STRATEGY_LOADERS["hybrid"] = [
            fake_load_indexed_chunk_sets,
            fake_load_bm25_index_chunk_sets,
        ]


def _patch_registry_path(monkeypatch, registry_file: Path) -> None:
    """Redirect SnapshotPaths.from_paths_config so the loader finds our
    test registry instead of `snapshots/registry.json` in CWD.
    """
    from src.rag.core.snapshot.paths import Path as _Path
    from src.rag.core.snapshot.paths import SnapshotPaths as _SnapshotPaths

    def fake_from_paths_config(paths_cfg, versioned_name=None):
        return _SnapshotPaths(
            building=_Path("snapshots/building"),
            chunks=_Path("snapshots/building/chunks"),
            transformed_chunks=_Path("snapshots/building/transformed_chunks"),
            index=_Path("snapshots/building/index"),
            configs=_Path("snapshots/building/configs"),
            metadata=_Path("snapshots/building/metadata.json"),
            registry_file=registry_file,
            config_files=(),
        )

    monkeypatch.setattr(_SnapshotPaths, "from_paths_config", fake_from_paths_config)


def test_dense_snapshot_roundtrip(tmp_path: Path, monkeypatch):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=True, hybrid=False)

    folder_name = lifecycle.start_new_build(snapshots_dir, registry_file, "dense", dataset_key="test-dataset")
    snapshot = snapshots_dir / folder_name
    _seed_snapshot(snapshot, "dense")

    lifecycle.finalise_snapshot(
        paths=_paths_for(snapshot),
        registry_file=registry_file,
        versioned_name=folder_name,
    )

    loaded = loader.load_snapshot(snapshot)
    assert loaded.retrieval_strategy == "dense"
    assert len(loaded.indexed_chunk_sets) == 1
    assert loaded.indexed_chunk_sets[0].name == "whole_recipe"
    assert loaded.bm25_indexed_chunk_sets == []
    assert loaded.base_config == {"rag": {"strategy": "dense"}}


def test_hybrid_snapshot_roundtrip(tmp_path: Path, monkeypatch):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=False, hybrid=True)

    folder_name = lifecycle.start_new_build(snapshots_dir, registry_file, "hybrid", dataset_key="test-dataset")
    snapshot = snapshots_dir / folder_name
    _seed_snapshot(snapshot, "hybrid")

    lifecycle.finalise_snapshot(
        paths=_paths_for(snapshot),
        registry_file=registry_file,
        versioned_name=folder_name,
    )

    loaded = loader.load_snapshot(snapshot)
    assert loaded.retrieval_strategy == "hybrid"
    assert {cs.name for cs in loaded.indexed_chunk_sets} == {"dense"}
    assert {cs.name for cs in loaded.bm25_indexed_chunk_sets} == {"sparse"}


def test_load_refuses_non_complete_snapshot(tmp_path: Path, monkeypatch):
    """Registry hygiene: loader must refuse anything whose status is not
    'complete'. This is what prevents a half-built snapshot from being
    served to users."""
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=True, hybrid=False)

    folder_name = lifecycle.start_new_build(snapshots_dir, registry_file, "dense", dataset_key="test-dataset")
    snapshot = snapshots_dir / folder_name
    _seed_snapshot(snapshot, "dense")
    # No finalise_snapshot - the registry still has status='building'.

    with pytest.raises(RuntimeError, match="status is 'building'"):
        loader.load_snapshot(snapshot)


def _paths_for(folder: Path):
    """Build a SnapshotPaths manually so we don't need the global paths
    config to resolve during a test."""
    from src.rag.core.snapshot.paths import SnapshotPaths

    return SnapshotPaths(
        building=folder,
        chunks=folder / "chunks",
        transformed_chunks=folder / "transformed_chunks",
        index=folder / "index",
        configs=folder / "configs",
        metadata=folder / "metadata.json",
        registry_file=folder.parent / "registry.json",
        config_files=(),
    )


