from dataclasses import replace
from pathlib import Path

import pytest

from src.rag.core.snapshot import lifecycle, loader
from src.rag.core.snapshot import paths as paths_module
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.retrieval.dense.indexer import IndexedChunkSet
from src.rag.retrieval.sparse.bm25indexer import BM25IndexedChunkSet


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)


def _seed_configs(source_dir: Path, strategy: str) -> None:
    """Write the configs the seal copies into the versioned folder.

    A build writes its configs into the staging tree, so a test that wants a
    loadable snapshot lays them down there before finalising.
    """
    configs_dir = source_dir / "configs"
    _write_yaml(configs_dir / "base_config.yaml", {"rag": {"strategy": strategy}})
    _write_yaml(configs_dir / "prompts_config.yaml", {"personas": {}})


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


def _patch_snapshots_root(monkeypatch, root: Path) -> None:
    """Send the sealed folders a test seals into tmp_path.

    `SnapshotPaths.snapshots_root` reads this module-level root, so patching it
    keeps a test's seals out of the repository's own `snapshots/`.
    """
    monkeypatch.setattr(paths_module, "_SNAPSHOTS_ROOT", root)


def _patch_registry_path(monkeypatch, registry_file: Path) -> None:
    """Redirect SnapshotPaths.from_paths_config so the loader finds our
    test registry instead of `snapshots/registry.json` in CWD.
    """
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


def test_dense_snapshot_roundtrip(tmp_path: Path, monkeypatch):
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=True, hybrid=False)

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="test-dataset"
    )
    _seed_configs(staging, "dense")

    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)
    sealed = snapshots_root / name
    assert sealed.is_dir()
    # The seal copies, so the outputs DVC declared for the build stay in place.
    assert staging.is_dir()

    loaded = loader.load_snapshot(sealed)
    assert loaded.retrieval_strategy == "dense"
    assert len(loaded.indexed_chunk_sets) == 1
    assert loaded.indexed_chunk_sets[0].name == "whole_recipe"
    assert loaded.bm25_indexed_chunk_sets == []
    assert loaded.base_config == {"rag": {"strategy": "dense"}}


def test_hybrid_snapshot_roundtrip(tmp_path: Path, monkeypatch):
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=False, hybrid=True)

    lifecycle.start_new_build(
        staging_dir=staging, strategy="hybrid", dataset_key="test-dataset"
    )
    _seed_configs(staging, "hybrid")

    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)
    sealed = snapshots_root / name

    loaded = loader.load_snapshot(sealed)
    assert loaded.retrieval_strategy == "hybrid"
    assert {cs.name for cs in loaded.indexed_chunk_sets} == {"dense"}
    assert {cs.name for cs in loaded.bm25_indexed_chunk_sets} == {"sparse"}


def test_every_seal_claims_its_own_folder(tmp_path: Path, monkeypatch):
    """A build that changes nothing still seals a folder of its own.

    Sealing copies the staging tree into a name claimed at that moment, so
    finalising the same bytes twice leaves two folders, each readable on its
    own. Reusing a name would mean the second build's contents decided what the
    first build's folder served.
    """
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=True, hybrid=False)

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="test-dataset"
    )
    _seed_configs(staging, "dense")

    first = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)
    name = lifecycle.finalise_snapshot(_staging_paths(staging, registry_file), registry_file)

    assert name != first
    for folder in (first, name):
        assert loader.load_snapshot(snapshots_root / folder).retrieval_strategy == "dense"


def test_load_refuses_an_unsealed_staging_tree(tmp_path: Path, monkeypatch):
    """Registry hygiene: the loader must refuse anything the registry has not
    marked 'complete'. A staging tree in flight carries no status at all, which
    is what keeps a half-built build from being served to users."""
    staging = tmp_path / "build" / "staging"
    snapshots_root = tmp_path / "snapshots"
    registry_file = snapshots_root / "registry.json"
    _patch_snapshots_root(monkeypatch, snapshots_root)
    _patch_registry_path(monkeypatch, registry_file)
    _patch_loaders(monkeypatch, dense=True, hybrid=False)

    lifecycle.start_new_build(
        staging_dir=staging, strategy="dense", dataset_key="test-dataset"
    )
    _seed_configs(staging, "dense")
    # No finalise_snapshot, so the registry carries no status for this tree.

    with pytest.raises(RuntimeError, match="registry status is None"):
        loader.load_snapshot(staging)
