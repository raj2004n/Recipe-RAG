"""Snapshot lifecycle.

Archives any existing 'latest' snapshot to its id
Copies build configs into snapshot
Stamps metadata with snapshot_id and created_at
Renmaes 'building' (current snapshot) to 'latest'
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .paths import SnapshotPaths


def save_config(save_path: Path, config_path: str) -> None:
    """Copy a config file into a snapshot's configs directory.

    'save_path' must point at an existing or creatable directory. 
    The source filename is preserved as the destination filename.
    """
    save_dir = Path(save_path)
    src = Path(config_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, save_dir / src.name)


def _archive_existing_latest(paths: SnapshotPaths) -> None:
    """Move `snapshots/latest` to 'snapshots/<snapshot_id>' if it exists.

    If 'paths.latest' does not exist than does nothing.
    
    If target id is already taken, then add timestap suffix.
    This would never be the case if an id is automatically generated (default),
    unless a snapshot is saved with custom id.
    """
    if not paths.latest.exists():
        return

    metadata_file = paths.latest / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No metadata file found at {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    snapshot_id = metadata["snapshot_id"]
    archive_path = paths.latest.parent / snapshot_id

    if archive_path.exists():
        suffix = time.strftime("%Y%m%d%H%M%S")
        archive_path = paths.latest.parent / f"{snapshot_id}_{suffix}"
        print(
            f"Warning: snapshot with id {snapshot_id} already exists; "
            f"archiving latest to {archive_path.name}"
        )

    os.rename(paths.latest, archive_path)


def _copy_configs_into(paths: SnapshotPaths) -> None:
    """Copy each config listed in `paths.config_files` into the snapshot.

    Source paths are resolved at `SnapshotPaths` construction time from
    `PATHS_CONFIG["configs"]` — see `configs/paths_config.yaml`.
    """
    for config_path in paths.config_files:
        save_config(save_path=paths.configs, config_path=config_path)


def finalise_snapshot(
    paths: SnapshotPaths,
    snapshot_id: str | None = None,
    created_at: str | None = None,
) -> None:
    """Seal the 'building' snapshot into 'latest'."""
    if not paths.building.exists():
        raise FileNotFoundError(f"Building directory not found at {paths.building}")

    _archive_existing_latest(paths)
    _copy_configs_into(paths)

    snapshot_id = snapshot_id or str(uuid.uuid4())
    created_at = created_at or time.strftime("%Y-%m-%d %H:%M:%S")

    with open(paths.metadata, "r") as f:
        metadata = json.load(f)

    metadata.update({
        "snapshot_id": snapshot_id,
        "created_at": created_at,
    })

    with open(paths.metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    # finalise by renaming 'building' to 'latest'
    paths.building.rename(paths.latest)
