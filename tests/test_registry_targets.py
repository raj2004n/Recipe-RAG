import contextlib
import json
import types
from pathlib import Path

from scripts import build_rag
from src.rag.core.snapshot import lifecycle, registry
from src.rag.core.snapshot.paths import SnapshotPaths


# --- reading a registry written before the rename -------------------------


def test_old_target_names_read_under_their_new_names(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"latest": "hybrid_1", "baseline": "hybrid_0"},
        "status": {"hybrid_0": "complete", "hybrid_1": "complete"},
    }))

    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_1"
    assert registry.get_folder_for_target(registry_file, "serve") == "hybrid_0"

    a = registry.get_folder_for_target(registry_file, "latest")
    b = registry.get_folder_for_target(registry_file, "baseline")
    assert (a, b) == (None, None)


def test_a_legacy_current_key_is_dropped(tmp_path):
    """`current` named the folder a running build wrote into.

    The staging tree replaced that folder, so a migrated registry does not
    carry the key forward under a name nothing reads.
    """
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1"},
        "status": {"hybrid_1": "complete"},
    }))

    assert registry.get_folder_for_target(registry_file, "current") is None
    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_1"


def test_the_next_write_persists_the_new_names(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "hybrid_1", "latest": "hybrid_1", "baseline": "hybrid_0"},
        "status": {"hybrid_0": "complete", "hybrid_1": "complete"},
    }))

    registry.update_status(registry_file, "hybrid_1", "complete")

    on_disk = json.loads(registry_file.read_text())
    assert on_disk["targets"] == {"latest_build": "hybrid_1", "serve": "hybrid_0"}


def test_a_key_already_carrying_its_new_name_wins(tmp_path):
    """A registry holding both spellings resolves the same way on every read."""
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"latest": "hybrid_1", "latest_build": "hybrid_2"},
        "status": {},
    }))

    assert registry.get_folder_for_target(registry_file, "latest_build") == "hybrid_2"


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
    """The run's recording step reads `latest_build` and describes that folder.

    Every stage has run by the time the recording step reads the registry, so
    the snapshot it records is the one `finalise_build` copied the staging tree
    into. The params and metrics below are the ones the guard around the
    recording step would swallow if that lookup came back empty.
    """
    monkeypatch.chdir(tmp_path)
    snapshots_root = Path("snapshots")
    registry_file = snapshots_root / "registry.json"
    staging = Path("build/staging")

    lifecycle.start_new_build(
        staging_dir=staging, strategy="hybrid", dataset_key="test-dataset"
    )
    (staging / "chunks").mkdir(parents=True, exist_ok=True)
    (staging / "index").mkdir(parents=True, exist_ok=True)
    (staging / "chunks" / "chunk_sets.json").write_text(json.dumps([
        {"name": "sparse", "chunks": [{"chunk_id": 0}, {"chunk_id": 1}]},
        {"name": "dense", "chunks": [{"chunk_id": 2}]},
    ]))
    (staging / "index" / "indexed_chunk_sets.json").write_text(json.dumps([{"name": "dense"}]))
    (staging / "index" / "bm25_index_chunk_sets.json").write_text(json.dumps([{"name": "sparse"}]))

    # The paths a stage resolves, which is what the build run reads after the
    # seal: staging under build/, the registry under snapshots/.
    paths = SnapshotPaths.from_paths_config({})
    name = lifecycle.finalise_snapshot(paths, registry_file)
    assert registry.get_folder_for_target(registry_file, "latest_build") == name

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
    assert str(snapshots_root / name / "metadata.json") in fake.artifacts
    assert "could not be recorded" not in caplog.text
