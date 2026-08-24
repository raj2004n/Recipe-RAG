"""Tests for the LocalDatasetResolver.

The resolver reads the dataset entry from the registry, verifies the file
exists, and verifies its checksum still matches. Failures are surfaced as
exceptions so the FastAPI lifespan can fail closed.
"""

import json
from pathlib import Path

import pytest

from src.rag.core.datasets.resolver import (
    LocalDatasetResolver,
    MissingDatasetError,
    ChecksumMismatchError,
)
from src.rag.core.snapshot import registry


def _seed_registry(registry_file: Path, key: str, path: Path, checksum: str | None) -> None:
    """Write a minimal registry with one dataset entry."""
    entry = {"path": str(path), "created_at": "2026-08-23 12:00:00", "size_bytes": 0}
    if checksum is not None:
        entry["checksum"] = checksum
    payload = {"targets": {}, "status": {}, "datasets": {key: entry}}
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_file, "w") as f:
        json.dump(payload, f)


def _write_dataset_file(path: Path, contents: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    import hashlib
    return f"sha256:{hashlib.sha256(contents).hexdigest()}"


def test_resolve_happy_path(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    checksum = _write_dataset_file(dataset_path, b'{"recipes": []}')
    _seed_registry(registry_file, "recipes_v1", dataset_path, checksum)

    resolver = LocalDatasetResolver(registry_file, registry.get_dataset)
    resolved = resolver.resolve("recipes_v1")

    assert resolved.key == "recipes_v1"
    assert resolved.path == dataset_path
    assert resolved.checksum == checksum
    assert resolved.source == "local"
    assert resolver.exists("recipes_v1") is True


def test_resolve_missing_key(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    _seed_registry(registry_file, "other", tmp_path / "x", "sha256:deadbeef")

    resolver = LocalDatasetResolver(registry_file, registry.get_dataset)

    with pytest.raises(MissingDatasetError, match="not registered"):
        resolver.resolve("nonexistent")

    assert resolver.exists("nonexistent") is False


def test_resolve_file_gone(tmp_path: Path):
    """A registered dataset whose file has been deleted must fail closed."""
    registry_file = tmp_path / "registry.json"
    missing_path = tmp_path / "deleted.json"
    _seed_registry(registry_file, "gone", missing_path, "sha256:deadbeef")

    resolver = LocalDatasetResolver(registry_file, registry.get_dataset)

    with pytest.raises(MissingDatasetError, match="missing"):
        resolver.resolve("gone")


def test_resolve_checksum_mismatch(tmp_path: Path):
    """A file that has changed since registration must fail closed.

    This is the protection against the dataset being silently swapped
    between when the snapshot was built and when the service starts.
    """
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"new contents")
    _seed_registry(registry_file, "rotated", dataset_path, "sha256:aaaa1111")

    resolver = LocalDatasetResolver(registry_file, registry.get_dataset)

    with pytest.raises(ChecksumMismatchError, match="checksum"):
        resolver.resolve("rotated")


def test_resolve_entry_without_path(tmp_path: Path):
    """An entry with no `path` (placeholder for cloud-backed datasets)
    must fail rather than returning a partial ResolvedDataset."""
    registry_file = tmp_path / "registry.json"
    payload = {
        "targets": {},
        "status": {},
        "datasets": {
            "cloud_only": {"uri": "s3://bucket/key", "checksum": "sha256:abcd"}
        },
    }
    with open(registry_file, "w") as f:
        json.dump(payload, f)

    resolver = LocalDatasetResolver(registry_file, registry.get_dataset)

    with pytest.raises(MissingDatasetError, match="no `path`"):
        resolver.resolve("cloud_only")
