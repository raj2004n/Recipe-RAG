import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.snapshot import lifecycle, loader
from src.snapshot import paths as paths_module
from src.snapshot.paths import SnapshotPaths
from src.build.indexer import IndexedChunkSet
from src.build.bm25indexer import BM25IndexedChunkSet


def _seed_configs(source_dir: Path, strategy: str) -> None:
    """Write the configs the seal copies into the versioned folder.

    Metadata is left alone: `start_new_build` wrote it, and clobbering it here
    would erase the dataset_key pinned at set_strategy, which is exactly what
    the rest of these tests are checking.
    """
    configs_dir = source_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "base_config.yaml").write_text(f"rag:\n  strategy: {strategy}\n")
    (configs_dir / "prompts_config.yaml").write_text("personas: {}\n")


def _patch_loaders(monkeypatch, *, strategy: str) -> None:
    def fake_load_indexed_chunk_sets(save_path, chunks_save_path=None):
        if strategy == "dense":
            return [IndexedChunkSet(name="whole_recipe", index=None, chunks={})], {
                "whole_recipe": {0: None}
            }
        return [IndexedChunkSet(name="dense", index=None, chunks={})], {"dense": {0: None}}

    def fake_load_bm25_index_chunk_sets(save_path, chunks_save_path=None):
        return [BM25IndexedChunkSet(name="sparse", index=None, chunks={})], {
            "sparse": {0: None}
        }

    monkeypatch.setattr(loader, "load_indexed_chunk_sets", fake_load_indexed_chunk_sets)
    monkeypatch.setattr(loader, "load_bm25_index_chunk_sets", fake_load_bm25_index_chunk_sets)
    loader._STRATEGY_LOADERS["dense"] = [fake_load_indexed_chunk_sets]
    loader._STRATEGY_LOADERS["hybrid"] = [
        fake_load_indexed_chunk_sets,
        fake_load_bm25_index_chunk_sets,
    ]


def _patch_snapshots_root(monkeypatch, root: Path) -> None:
    """Send the sealed folders a test seals into tmp_path."""
    monkeypatch.setattr(paths_module, "_SNAPSHOTS_ROOT", root)


def _patch_registry_path(monkeypatch, registry_file: Path) -> None:
    """Redirect SnapshotPaths.from_paths_config so the loader finds our
    test registry instead of `snapshots/registry.json` in CWD."""
    real_from_paths_config = SnapshotPaths.from_paths_config

    def fake_from_paths_config(paths_cfg, versioned_name=None):
        resolved = real_from_paths_config(paths_cfg, versioned_name)
        return replace(resolved, registry_file=registry_file)

    monkeypatch.setattr(SnapshotPaths, "from_paths_config", fake_from_paths_config)


def _staging_paths(staging_dir: Path, registry_file: Path) -> SnapshotPaths:
    """The staging tree's paths, in the shape a build stage resolves them."""
    return SnapshotPaths(
        building=staging_dir,
        chunks=staging_dir / "chunks",
        transformed_chunks=staging_dir / "transformed_chunks",
        index=staging_dir / "index",
        configs=staging_dir / "configs",
        metadata=staging_dir / "metadata.json",
        registry_file=registry_file,
        config_files=(),
    )


def test_start_new_build_writes_dataset_key_at_set_strategy(tmp_path: Path):
    staging = tmp_path / "build" / "staging"

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="recipes_v3"
    )
    metadata = json.loads((staging / "metadata.json").read_text())

    assert metadata == {"strategy": "dense", "dataset_key": "recipes_v3"}


def test_start_new_build_without_dataset_key_writes_only_strategy(tmp_path: Path):
    """The dataset_key arg is required for new builds. Without it the
    metadata only carries the strategy - and load_snapshot will fail to
    load that snapshot because the dataclass field is required."""
    staging = tmp_path / "build" / "staging"

    lifecycle.start_new_build(staging_dir=staging, strategy="dense")
    metadata = json.loads((staging / "metadata.json").read_text())

    assert metadata == {"strategy": "dense"}


def test_finalise_preserves_dataset_key(tmp_path: Path, monkeypatch):
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="recipes_v3"
    )

    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)

    metadata = json.loads((snapshots_root / name / "metadata.json").read_text())
    assert metadata["dataset_key"] == "recipes_v3"
    assert metadata["strategy"] == "dense"
    assert "created_at" in metadata


def test_load_snapshot_surfaces_dataset_key(tmp_path: Path, monkeypatch):
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, strategy="dense")

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="recipes_v3"
    )
    _seed_configs(staging, "dense")

    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)

    loaded = loader.load_snapshot(snapshots_root / name)
    assert loaded.dataset_key == "recipes_v3"


def test_load_snapshot_raises_when_dataset_key_missing(tmp_path: Path, monkeypatch):
    """No silent defaults. A snapshot built without a dataset_key cannot
    be loaded because the dataclass field is required."""
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, strategy="dense")

    lifecycle.start_new_build(staging_dir=staging, strategy="dense")
    _seed_configs(staging, "dense")
    # Re-write metadata without the key to simulate a legacy snapshot.
    (staging / "metadata.json").write_text(json.dumps({"strategy": "dense"}))

    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)

    with pytest.raises(KeyError):
        loader.load_snapshot(snapshots_root / name)
