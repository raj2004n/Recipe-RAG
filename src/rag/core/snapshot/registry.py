"""Registry management for RAG snapshots.

Handles the naming of versioned builds and the maintenance of the registry.json
mapping that determines which snapshot is currently active.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, NotRequired


class RegistryData(TypedDict, total=False):
    targets: dict[str, str]  # target_name -> folder_name (e.g. "latest_build" -> "dense_2026_0")
    status: dict[str, str]   # folder_name -> status ("building" | "complete" | "abandoned")
    promoted: dict[str, "PromotionRecord"]  # folder_name -> record of its promotion


class DatasetEntry(TypedDict, total=False):
    """A registered dataset. Either `path` or `uri` must be set.

    `path` is a project-relative or absolute local path; `uri` is reserved for
    cloud backends (S3, GCS) that arrive in a later step.
    """
    path: str
    uri: str
    created_at: str
    checksum: str   # "sha256:<hex>"
    size_bytes: int


class PromotionRecord(TypedDict):
    """What the promotion gate recorded when a snapshot became the baseline.

    Carries the identity of the end-to-end MLflow run the verdict came from and
    the per-check outcome, so a promotion is auditable from the registry alone.
    `verdict` is the gate's own outcome for that run, "pass" or "fail"; it is
    what marks a promotion that went ahead against the bars. A record written
    before the gate could record an override has no verdict key and reads as a
    pass.
    """
    run_id: str
    run_name: str
    promoted_at: str
    checks: list
    verdict: NotRequired[str]


# The target keys as they were named before the names described their readers.
_TARGET_RENAMES = {
    "current": "build_in_progress",
    "latest": "latest_build",
    "baseline": "serve",
}


def _migrate_legacy(data: dict) -> dict:
    """Convert a legacy {current, status} registry into the {targets, status} shape.

    The legacy shape stored a single current target and a single status string.
    The new shape is a target -> folder dict plus a folder -> status dict.
    Idempotent: returns the new shape unchanged if already migrated.
    """
    if "targets" in data and "status" in data:
        return {
            "targets": dict(data["targets"]),
            "status": dict(data["status"]),
            "datasets": dict(data.get("datasets", {})),
            "promoted": dict(data.get("promoted", {})),
        }

    if "current" in data:
        folder = data["current"]
        status = data.get("status", "complete")
        return {
            "targets": {"current": folder, "latest": folder} if folder else {"current": folder},
            "status": {folder: status} if folder else {},
            "datasets": {},
            "promoted": {},
        }

    return {"targets": {}, "status": {}, "datasets": {}, "promoted": {}}


def _migrate_target_names(targets: dict) -> dict:
    """Rename the target keys a registry may still carry under their old names.

    The targets are named for what reads them: `build_in_progress` for the
    folder the active build writes into, `latest_build` for the newest sealed
    folder, and `serve` for the folder a human promoted. Registries written
    before those names hold `current`, `latest` and `baseline` instead, so
    reading maps them across and the next write persists the new spelling. A
    key already carrying its new name wins over a renamed one, so a registry
    holding both resolves the same way on every read.
    """
    renamed: dict[str, str] = {}
    for key, folder in targets.items():
        if key in _TARGET_RENAMES:
            renamed.setdefault(_TARGET_RENAMES[key], folder)
        else:
            renamed[key] = folder
    return renamed


def claim_snapshot_name(snapshots_dir: Path, strategy: str) -> str:
    """Atomically reserve a unique snapshot name and create its folder.

    Closes the TOCTOU race in name generation by attempting to create the
    folder directly; a `FileExistsError` bumps the index and retries.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = f"{strategy}_{timestamp}"
    index = 0
    while True:
        name = f"{prefix}_{index}"
        target = snapshots_dir / name
        try:
            target.mkdir(exist_ok=False)
        except FileExistsError:
            index += 1
            continue
        return name


def _read_registry(registry_path: Path) -> RegistryData:
    """Read the registry file, migrating legacy shapes to the current one.
    """
    if not registry_path.exists():
        return {"targets": {}, "status": {}, "datasets": {}, "promoted": {}}

    try:
        with open(registry_path, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"targets": {}, "status": {}, "datasets": {}, "promoted": {}}

    if not isinstance(raw, dict):
        return {"targets": {}, "status": {}, "datasets": {}, "promoted": {}}

    migrated = _migrate_legacy(raw)
    return {
        "targets": _migrate_target_names(dict(migrated.get("targets", {}))),
        "status": dict(migrated.get("status", {})),
        "datasets": dict(migrated.get("datasets", {})),
        "promoted": dict(migrated.get("promoted", {})),
    }


def _write_registry(registry_path: Path, data: RegistryData) -> None:
    """Write registry data atomically.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = registry_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4)
    temp_path.replace(registry_path)


def update_target(registry_path: Path, target: str, folder: str) -> None:
    """Map a target (e.g. 'latest_build') to a specific snapshot folder.
    """
    data = _read_registry(registry_path)
    data["targets"][target] = folder
    _write_registry(registry_path, data)


def clear_target(registry_path: Path, target: str, folder: str | None = None) -> None:
    """Drop a target from the registry.

    Passing `folder` guards the drop: the target is cleared only while it still
    names that folder, so a caller releasing its own claim cannot take away a
    newer one that has since been written.
    """
    data = _read_registry(registry_path)
    if target not in data["targets"]:
        return
    if folder is not None and data["targets"][target] != folder:
        return
    del data["targets"][target]
    _write_registry(registry_path, data)


def update_status(registry_path: Path, folder: str, status: str) -> None:
    """Update the build status of a snapshot folder.
    """
    data = _read_registry(registry_path)
    data["status"][folder] = status
    _write_registry(registry_path, data)


def record_promotion(
    registry_path: Path,
    folder: str,
    *,
    run_id: str,
    run_name: str,
    verdict: str,
    checks: list,
    promoted_at: str | None = None,
) -> None:
    """Record a snapshot as the one to serve together with what the gate saw.

    Sets targets["serve"] to the folder and stores a promoted record keyed by
    the folder name carrying the governing MLflow run's identity, the gate's
    verdict and its per-check outcome, so a promotion is auditable from the
    registry alone and a promotion that went ahead against the bars is
    distinguishable from one that cleared them. Re-promoting the same folder
    overwrites its record in place; promoting a different folder keeps earlier
    promoted records intact. Writes atomically like the other registry helpers.
    """
    data = _read_registry(registry_path)
    data["targets"]["serve"] = folder
    promoted = data.setdefault("promoted", {})
    promoted[folder] = {
        "run_id": run_id,
        "run_name": run_name,
        "promoted_at": promoted_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "checks": checks,
    }
    _write_registry(registry_path, data)


def get_promotion_record(registry_path: Path, folder: str) -> Optional[PromotionRecord]:
    """The recorded promotion for a folder, or None if it was never promoted."""
    return _read_registry(registry_path)["promoted"].get(folder)


def get_folder_for_target(registry_path: Path, target: str) -> Optional[str]:
    """Resolve a target name to its associated snapshot folder.
    """
    data = _read_registry(registry_path)
    return data["targets"].get(target)


def get_status_for_folder(registry_path: Path, folder: str) -> Optional[str]:
    """Get the build status of a snapshot folder.
    """
    data = _read_registry(registry_path)
    return data["status"].get(folder)


def list_status_for_folders(registry_path: Path) -> dict[str, str]:
    """Return the folder -> status mapping.
    """
    return _read_registry(registry_path)["status"]


def get_dataset(registry_path: Path, key: str) -> Optional[DatasetEntry]:
    """Return a dataset entry by key, or None if not registered.
    """
    return _read_registry(registry_path)["datasets"].get(key)


def list_datasets(registry_path: Path) -> dict[str, DatasetEntry]:
    """Return the full datasets section as a key -> entry mapping.
    """
    return dict(_read_registry(registry_path)["datasets"])


def upsert_dataset(registry_path: Path, key: str, entry: DatasetEntry) -> None:
    """Insert or replace a dataset entry by key.

    Atomic write, same as the other registry helpers.
    """
    data = _read_registry(registry_path)
    data["datasets"][key] = entry
    _write_registry(registry_path, data)


def remove_dataset(registry_path: Path, key: str) -> bool:
    """Remove a dataset entry by key. Returns True if it existed.
    """
    data = _read_registry(registry_path)
    if key not in data["datasets"]:
        return False
    del data["datasets"][key]
    _write_registry(registry_path, data)
    return True


def sweep_stale_building_builds(
    snapshots_dir: Path,
    registry_path: Path,
    max_age_hours: int = 24,
    required_subdirs: tuple[str, ...] = ("chunks", "index"),
) -> list[str]:
    """Mark stale 'building'-status folders as 'abandoned' in the registry.

    A build is considered stale if its `status` is `"building"` AND one of:
        - the build folder is missing the required subdirs (e.g. `chunks/`),
        - the build folder's `metadata.json` mtime is older than `max_age_hours`.

    Abandoning the folder also releases `build_in_progress` when that target
    still names it, so the target only ever points at a build that is running.

    Returns the list of folder names that were flipped to `"abandoned"`.
    """
    data = _read_registry(registry_path)
    now = time.time()
    cutoff = now - max_age_hours * 3600
    swept: list[str] = []

    for folder, status in list(data["status"].items()):
        if status != "building":
            continue

        folder_path = snapshots_dir / folder
        if not folder_path.is_dir():
            data["status"][folder] = "abandoned"
            swept.append(folder)
            continue

        missing_subdir = any(not (folder_path / sub).is_dir() for sub in required_subdirs)
        metadata = folder_path / "metadata.json"
        stale_mtime = metadata.exists() and metadata.stat().st_mtime < cutoff

        if missing_subdir or stale_mtime:
            data["status"][folder] = "abandoned"
            swept.append(folder)

    if swept:
        for folder in swept:
            if data["targets"].get("build_in_progress") == folder:
                del data["targets"]["build_in_progress"]
        _write_registry(registry_path, data)

    return swept
