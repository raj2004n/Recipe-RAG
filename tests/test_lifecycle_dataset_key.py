"""Tests for the dataset_key thread through the snapshot lifecycle.

Verifies that:
- start_new_build writes dataset_key into metadata.json at set_strategy
  (not at finalise), so the build is bound to the dataset before any
  chunking happens.
- finalise_snapshot preserves the key (does not overwrite it).
- load_snapshot surfaces it on Snapshot.dataset_key.
- omitting dataset_key is rejected: the dataclass field is required and
  metadata without the key raises KeyError on load. No silent defaults.
"""

import json
from pathlib import Path

from src.rag.core.snapshot import lifecycle, loader
from src.rag.retrieval.dense.indexer import IndexedChunkSet
from src.rag.retrieval.sparse.bm25indexer import BM25IndexedChunkSet


def _seed_minimum_files(snapshot_dir: Path, strategy: str) -> None:
    """Lay down the empty chunks/index/configs subdirs finalise expects.

    Metadata is left alone - `start_new_build` writes it. Clobbering it
    here would erase the dataset_key pinned at set_strategy, which is
    exactly what the rest of these tests are checking.
    """
    (snapshot_dir / "chunks").mkdir()
    (snapshot_dir / "index").mkdir()
    configs_dir = snapshot_dir / "configs"
    configs_dir.mkdir()
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


def _patch_registry_path(monkeypatch, registry_file: Path) -> None:
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


def _paths_for(folder: Path):
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


def test_start_new_build_writes_dataset_key_at_set_strategy(tmp_path: Path):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"

    name = lifecycle.start_new_build(
        snapshots_dir, registry_file, "dense", dataset_key="recipes_v3"
    )
    metadata = json.loads((snapshots_dir / name / "metadata.json").read_text())

    assert metadata == {"strategy": "dense", "dataset_key": "recipes_v3"}


def test_start_new_build_without_dataset_key_writes_only_strategy(tmp_path: Path):
    """The dataset_key arg is required for new builds. Without it the
    metadata only carries the strategy - and load_snapshot will fail to
    load that snapshot because the dataclass field is required."""
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"

    name = lifecycle.start_new_build(snapshots_dir, registry_file, "dense")
    metadata = json.loads((snapshots_dir / name / "metadata.json").read_text())

    assert metadata == {"strategy": "dense"}


def test_finalise_preserves_dataset_key(tmp_path: Path, monkeypatch):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)

    folder_name = lifecycle.start_new_build(
        snapshots_dir, registry_file, "dense", dataset_key="recipes_v3"
    )
    snapshot = snapshots_dir / folder_name
    _seed_minimum_files(snapshot, "dense")

    lifecycle.finalise_snapshot(
        paths=_paths_for(snapshot),
        registry_file=registry_file,
        versioned_name=folder_name,
    )

    metadata = json.loads((snapshot / "metadata.json").read_text())
    assert metadata["dataset_key"] == "recipes_v3"
    assert metadata["strategy"] == "dense"
    assert "created_at" in metadata


def test_load_snapshot_surfaces_dataset_key(tmp_path: Path, monkeypatch):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, strategy="dense")

    folder_name = lifecycle.start_new_build(
        snapshots_dir, registry_file, "dense", dataset_key="recipes_v3"
    )
    snapshot = snapshots_dir / folder_name
    _seed_minimum_files(snapshot, "dense")

    lifecycle.finalise_snapshot(
        paths=_paths_for(snapshot),
        registry_file=registry_file,
        versioned_name=folder_name,
    )

    loaded = loader.load_snapshot(snapshot)
    assert loaded.dataset_key == "recipes_v3"


def test_load_snapshot_raises_when_dataset_key_missing(tmp_path: Path, monkeypatch):
    """No silent defaults. A snapshot built without a dataset_key cannot
    be loaded because the dataclass field is required."""
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, strategy="dense")

    folder_name = lifecycle.start_new_build(snapshots_dir, registry_file, "dense")
    snapshot = snapshots_dir / folder_name
    _seed_minimum_files(snapshot, "dense")
    # Re-write metadata without the key to simulate a legacy snapshot.
    (snapshot / "metadata.json").write_text(json.dumps({"strategy": "dense"}))

    lifecycle.finalise_snapshot(
        paths=_paths_for(snapshot),
        registry_file=registry_file,
        versioned_name=folder_name,
    )

    import pytest
    with pytest.raises(KeyError):
        loader.load_snapshot(snapshot)
