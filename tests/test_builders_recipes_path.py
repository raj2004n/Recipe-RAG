"""Tests for the recipes_path threading through DenseBuilder and HybridBuilder.

Both builders used to read DATA_PATHS["cleaned_recipes"] directly, which
made it impossible to build against anything but the configured default.
Commit 2 lifts that: each builder takes the path as an argument and logs
its dataset identity at stage start.

These tests stub out the chunker so we don't need a real recipes file —
the contract under test is the signature and the logging, not the
chunking behaviour.
"""

import json
import logging
from pathlib import Path

import pytest

from src.rag.builder.dense_builder import DenseBuilder
from src.rag.builder.hybrid_builder import HybridBuilder
from src.rag.builder.base_builder import _log_dataset_identity
from src.rag.core.snapshot.paths import SnapshotPaths


def _paths_for(folder: Path) -> SnapshotPaths:
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


def _seed_metadata(folder: Path, dataset_key: str = "test-dataset") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps({"strategy": "dense", "dataset_key": dataset_key})
    )


def test_dense_builder_chunk_takes_recipes_path(tmp_path: Path, monkeypatch, caplog):
    recipes = tmp_path / "recipes.json"
    recipes.write_text("[]")
    snapshot = tmp_path / "snapshots" / "dense_20260824_0000_0"
    paths = _paths_for(snapshot)
    _seed_metadata(snapshot)

    captured: dict = {}

    def fake_chunk_recipes(*, strategy, recipes_path):
        captured["strategy"] = strategy
        captured["recipes_path"] = recipes_path
        return []

    monkeypatch.setattr("src.rag.builder.dense_builder.chunk_recipes", fake_chunk_recipes)

    with caplog.at_level(logging.INFO):
        DenseBuilder().chunk(paths, recipes)

    assert captured == {"strategy": "whole_recipe", "recipes_path": recipes}
    # Identity log line emitted at chunk start.
    assert any("dataset_key=test-dataset" in r.message for r in caplog.records)
    assert any(str(recipes) in r.message for r in caplog.records)


def test_hybrid_builder_chunk_takes_recipes_path(tmp_path: Path, monkeypatch, caplog):
    recipes = tmp_path / "recipes.json"
    recipes.write_text("[]")
    snapshot = tmp_path / "snapshots" / "hybrid_20260824_0000_0"
    paths = _paths_for(snapshot)
    _seed_metadata(snapshot)

    captured: dict = {}

    def fake_chunk_recipes(*, strategy, recipes_path):
        captured["strategy"] = strategy
        captured["recipes_path"] = recipes_path
        return []

    monkeypatch.setattr("src.rag.builder.hybrid_builder.chunk_recipes", fake_chunk_recipes)

    with caplog.at_level(logging.INFO):
        HybridBuilder().chunk(paths, recipes)

    assert captured == {"strategy": "hybrid", "recipes_path": recipes}
    assert any("dataset_key=test-dataset" in r.message for r in caplog.records)


def test_log_dataset_identity_no_recipes_path_at_later_stages(tmp_path, caplog):
    """At transform/index we don't re-pass the recipes_path — the log
    line must still emit, drawing the key from metadata.json only."""
    snapshot = tmp_path / "snapshots" / "dense_x"
    paths = _paths_for(snapshot)
    _seed_metadata(snapshot, dataset_key="later-stage-key")

    with caplog.at_level(logging.INFO):
        _log_dataset_identity("transform", paths)

    assert any("dataset_key=later-stage-key" in r.message for r in caplog.records)
    # No checksum at later stages — that would imply re-reading the file.
    assert not any("checksum=" in r.message for r in caplog.records)


def test_builders_no_longer_import_DATA_PATHS():
    """Direct module check: the builders must not reach for DATA_PATHS."""
    import src.rag.builder.dense_builder as dense_mod
    import src.rag.builder.hybrid_builder as hybrid_mod

    # If either module imports DATA_PATHS it would show up as a bound
    # name in the module's namespace.
    assert not hasattr(dense_mod, "DATA_PATHS")
    assert not hasattr(hybrid_mod, "DATA_PATHS")
