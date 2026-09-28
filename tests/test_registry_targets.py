"""Tests for the registry targets and the names they now carry.

The targets are named for what reads them: `build_in_progress` for the folder
the active build writes into, `latest_build` for the newest sealed folder, and
`serve` for the folder a human promoted. These tests pin the two properties the
build and the service rely on: a target is released when the build it named is
no longer running, and a registry written before the rename still resolves.
"""

import contextlib
import json
import types
from pathlib import Path

from scripts import build_rag
from src.rag.core.snapshot import lifecycle, registry
from src.rag.core.snapshot.paths import SnapshotPaths


def _paths(snapshots_dir: Path, name: str) -> SnapshotPaths:
    """Build the paths for one snapshot folder under a test snapshots root."""
    folder = snapshots_dir / name
    return SnapshotPaths(
        building=folder,
        chunks=folder / "chunks",
        transformed_chunks=folder / "transformed_chunks",
        index=folder / "index",
        configs=folder / "configs",
        metadata=folder / "metadata.json",
        registry_file=snapshots_dir / "registry.json",
        config_files=(),
    )


# --- reading a registry written before the rename -------------------------


def test_old_target_names_read_under_their_new_names(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1", "baseline": "hybrid_0"},
        "status": {"hybrid_0": "complete", "hybrid_1": "building"},
    }))

    assert registry.get_folder_for_target(registry_file, "build_in_progress") == "hybrid_1"
    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_1"
    assert registry.get_folder_for_target(registry_file, "serve") == "hybrid_0"

    a = registry.get_folder_for_target(registry_file, "current")
    b = registry.get_folder_for_target(registry_file, "baseline")
    c = registry.get_folder_for_target(registry_file, "latest")
    assert (a, b, c) == (None, None, None)


def test_a_legacy_current_that_names_a_sealed_folder_is_dropped(tmp_path):
    """`current` was never released at seal, so a migrated one can be stale."""
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1"},
        "status": {"hybrid_1": "complete"},
    }))

    assert registry.get_folder_for_target(registry_file, "build_in_progress") is None
    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_1"


def test_a_build_in_progress_is_dropped_from_the_file_by_the_next_write(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1", "baseline": "hybrid_0"},
        "status": {"hybrid_0": "complete", "hybrid_1": "complete"},
    }))

    registry.update_status(registry_file, "hybrid_1", "complete")

    on_disk = json.loads(registry_file.read_text())
    assert on_disk["targets"] == {"latest_build": "hybrid_1", "serve": "hybrid_0"}


def test_a_running_build_in_progress_survives_a_read(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1"},
        "status": {"hybrid_1": "building"},
    }))

    assert registry.get_folder_for_target(registry_file, "build_in_progress") == "hybrid_1"


def test_the_next_write_persists_the_new_names(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1", "baseline": "hybrid_0"},
        "status": {"hybrid_0": "complete", "hybrid_1": "building"},
    }))

    registry.update_status(registry_file, "hybrid_1", "complete")

    on_disk = json.loads(registry_file.read_text())
    assert on_disk["targets"] == {"build_in_progress": "hybrid_1", "latest_build": "hybrid_1",
                                  "serve": "hybrid_0"}


def test_a_key_already_carrying_its_new_name_wins(tmp_path):
    """A registry holding both spellings resolves the same way on every read."""
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"latest": "hybrid_1", "latest_build": "hybrid_2"},
        "status": {},
    }))

    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_2"


# --- releasing build_in_progress -----------------------------------------


def test_a_new_build_leaves_a_resolvable_build_in_progress(tmp_path):
    """The claim a build makes survives its own writes.

    The claim is only kept while the folder it names is `building`, and the
    status and the target are two separate writes, so the order they go in
    decides whether the second write's read keeps the first write's claim.
    """
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    name = lifecycle.start_new_build(
        snapshots_dir=snapshots_dir,
        registry_file=registry_file,
        strategy="hybrid",
        dataset_key="test-dataset",
    )

    assert registry.get_folder_for_target(registry_file, "build_in_progress") == name
    assert registry.get_status_for_folder(registry_file, name) == "building"


def test_a_seal_clears_build_in_progress(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    name = lifecycle.start_new_build(
        snapshots_dir=snapshots_dir,
        registry_file=snapshots_dir / "registry.json",
        strategy="hybrid",
        dataset_key="test-dataset",
    )
    paths = _paths(snapshots_dir, name)
    (paths.chunks).mkdir(parents=True)
    (paths.index).mkdir(parents=True)

    lifecycle.finalise_snapshot(paths, paths.registry_file, name)

    assert registry.get_folder_for_target(paths.registry_file, "build_in_progress") is None
    assert registry.get_folder_for_target(paths.registry_file, "latest_build") == name
    assert registry.get_status_for_folder(paths.registry_file, name) == "complete"


def test_a_seal_leaves_a_build_in_progress_that_names_another_folder(tmp_path):
    """The release is guarded, so a build cannot drop a newer claim."""
    snapshots_dir = tmp_path / "snapshots"
    first = lifecycle.start_new_build(
        snapshots_dir=snapshots_dir,
        registry_file=snapshots_dir / "registry.json",
        strategy="hybrid",
        dataset_key="test-dataset",
    )
    registry_file = snapshots_dir / "registry.json"
    registry.update_status(registry_file, "hybrid_newer", "building")
    registry.update_target(registry_file, "build_in_progress", "hybrid_newer")

    paths = _paths(snapshots_dir, first)
    paths.chunks.mkdir(parents=True)
    paths.index.mkdir(parents=True)
    lifecycle.finalise_snapshot(paths, registry_file, first)

    assert registry.get_folder_for_target(registry_file, "build_in_progress") == "hybrid_newer"
    assert registry.get_folder_for_target(registry_file, "latest_build") == first


def test_a_sweep_clears_build_in_progress_only_for_the_folder_it_abandons(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    stale = "hybrid_20260101_0000_0"
    live = "hybrid_20260101_0000_1"
    # `stale` has no chunks/ subdir, so the sweep abandons it; `live` is intact.
    (snapshots_dir / stale).mkdir(parents=True)
    (snapshots_dir / stale / "metadata.json").write_text(json.dumps({"strategy": "hybrid"}))
    for sub in ("chunks", "index"):
        (snapshots_dir / live / sub).mkdir(parents=True)
    (snapshots_dir / live / "metadata.json").write_text(json.dumps({"strategy": "hybrid"}))

    registry.update_status(registry_file, stale, "building")
    registry.update_status(registry_file, live, "building")
    registry.update_target(registry_file, "build_in_progress", live)

    swept = registry.sweep_stale_building_builds(snapshots_dir, registry_file)

    assert sorted(swept) == [stale]
    assert registry.get_folder_for_target(registry_file, "build_in_progress") == live


def test_a_sweep_releases_build_in_progress_when_it_abandons_that_folder(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    registry_file = snapshots_dir / "registry.json"
    stale = "hybrid_20260101_0000_0"
    (snapshots_dir / stale).mkdir(parents=True)
    (snapshots_dir / stale / "metadata.json").write_text(json.dumps({"strategy": "hybrid"}))
    registry.update_status(registry_file, stale, "building")
    registry.update_target(registry_file, "build_in_progress", stale)

    swept = registry.sweep_stale_building_builds(snapshots_dir, registry_file)

    assert swept == [stale]
    assert registry.get_folder_for_target(registry_file, "build_in_progress") is None


# --- what the build run records after a seal ------------------------------


class _FakeMlflow:
    """Records what a build run logs, without a tracking store behind it."""

    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}
        self.artifacts: list[str] = []

    def set_tracking_uri(self, uri):
        pass

    def get_experiment_by_name(self, name):
        return None

    def create_experiment(self, name, artifact_location=None):
        pass

    def set_experiment(self, name):
        pass

    @contextlib.contextmanager
    def start_run(self, run_name=None):
        yield

    def log_params(self, params):
        self.params.update(params)

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_artifact(self, path):
        self.artifacts.append(path)


def test_a_build_run_records_the_sealed_snapshot(tmp_path, monkeypatch, caplog):
    """The run's recording step reads `latest_build`, not the released target.

    Sealing releases `build_in_progress`, so a recording step that still read
    that key would fail inside the guard that downgrades tracking failures to a
    warning: the build would be sealed and the run would hold no identity for
    it. The params and metrics below are the ones that guard would swallow.
    """
    monkeypatch.chdir(tmp_path)
    snapshots_dir = Path("snapshots")
    registry_file = snapshots_dir / "registry.json"

    name = lifecycle.start_new_build(
        snapshots_dir=snapshots_dir,
        registry_file=registry_file,
        strategy="hybrid",
        dataset_key="test-dataset",
    )
    paths = _paths(snapshots_dir, name)
    paths.chunks.mkdir(parents=True)
    paths.index.mkdir(parents=True)
    (paths.chunks / "chunk_sets.json").write_text(json.dumps([
        {"name": "sparse", "chunks": [{"chunk_id": 0}, {"chunk_id": 1}]},
        {"name": "dense", "chunks": [{"chunk_id": 2}]},
    ]))
    (paths.index / "indexed_chunk_sets.json").write_text(json.dumps([{"name": "dense"}]))
    (paths.index / "bm25_index_chunk_sets.json").write_text(json.dumps([{"name": "sparse"}]))
    lifecycle.finalise_snapshot(paths, registry_file, name)
    assert registry.get_folder_for_target(registry_file, "build_in_progress") is None

    fake = _FakeMlflow()
    monkeypatch.setattr(build_rag, "mlflow", fake)
    # The stages themselves are not under test here: what this run records is.
    monkeypatch.setattr(build_rag, "subprocess", types.SimpleNamespace(run=lambda *a, **k: None))

    with caplog.at_level("WARNING"):
        build_rag.build_rag()

    assert fake.params["snapshot_name"] == name
    assert fake.params["dataset_key"] == "test-dataset"
    assert fake.metrics["counts.chunks.sparse"] == 2
    assert fake.metrics["counts.chunks.dense"] == 1
    assert fake.metrics["counts.indexes.dense"] == 1
    assert fake.metrics["counts.indexes.bm25"] == 1
    assert str(paths.metadata) in fake.artifacts
    assert "could not be recorded" not in caplog.text
