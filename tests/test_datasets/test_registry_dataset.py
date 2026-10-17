import json
from pathlib import Path

import pytest

from src.snapshot.datasets import registry_dataset
from src.snapshot import registry


def _read_registry(registry_file: Path) -> dict:
    with open(registry_file) as f:
        return json.load(f)


def _write_dataset_file(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def test_register_new_dataset(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"recipe data")

    key = registry_dataset.register_dataset(registry_file, dataset_path)

    data = _read_registry(registry_file)
    assert key in data["datasets"]
    entry = data["datasets"][key]
    assert entry["path"] == str(dataset_path)
    assert entry["checksum"].startswith("sha256:")
    assert entry["size_bytes"] == len(b"recipe data")
    assert "created_at" in entry


def test_register_is_idempotent_for_same_file(tmp_path: Path):
    """Re-registering the same file must return the original key and not
    create a duplicate registry entry. This is what keeps many-to-one
    snapshots-datasets honest - a build that uses the same dataset twice
    registers it once."""
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"recipe data")

    key1 = registry_dataset.register_dataset(registry_file, dataset_path)
    key2 = registry_dataset.register_dataset(registry_file, dataset_path)

    assert key1 == key2
    data = _read_registry(registry_file)
    assert len(data["datasets"]) == 1


def test_register_different_file_gets_new_key(tmp_path: Path):
    """A different file at the same path gets a new key because its
    checksum differs. The registry stays honest: we don't reuse a key
    that was bound to a different physical file."""
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"version 1")
    key1 = registry_dataset.register_dataset(registry_file, dataset_path)

    _write_dataset_file(dataset_path, b"version 2")
    key2 = registry_dataset.register_dataset(registry_file, dataset_path)

    assert key1 != key2
    data = _read_registry(registry_file)
    assert len(data["datasets"]) == 2
    assert {key1, key2} == set(data["datasets"].keys())


def test_register_with_explicit_name(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"recipe data")

    key = registry_dataset.register_dataset(
        registry_file, dataset_path, name="my_pinned_key"
    )
    assert key == "my_pinned_key"
    assert "my_pinned_key" in _read_registry(registry_file)["datasets"]


def test_register_missing_file_raises(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(FileNotFoundError):
        registry_dataset.register_dataset(registry_file, missing)


def test_get_dataset_returns_entry(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    dataset_path = tmp_path / "recipes.json"
    _write_dataset_file(dataset_path, b"recipe data")
    key = registry_dataset.register_dataset(registry_file, dataset_path)

    entry = registry_dataset.get_dataset(registry_file, key)
    assert entry is not None
    assert entry["path"] == str(dataset_path)


def test_get_dataset_returns_none_for_unknown_key(tmp_path: Path):
    registry_file = tmp_path / "registry.json"
    assert registry_dataset.get_dataset(registry_file, "nope") is None


def test_registry_datasets_roundtrip_with_existing_keys(tmp_path: Path):
    """An existing registry with only `targets` and `status` (no `datasets`
    yet) should still load and report an empty datasets section. This is
    the migration story for any snapshot built before this step landed."""
    registry_file = tmp_path / "registry.json"
    payload = {
        "targets": {"latest": "dense_2026_0"},
        "status": {"dense_2026_0": "complete"},
    }
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_file, "w") as f:
        json.dump(payload, f)

    datasets = registry.list_datasets(registry_file)
    assert datasets == {}

    folder = registry.get_folder_for_target(registry_file, "latest_build")
    assert folder == "dense_2026_0"

    status = registry.get_status_for_folder(registry_file, folder)
    assert status == "complete"
