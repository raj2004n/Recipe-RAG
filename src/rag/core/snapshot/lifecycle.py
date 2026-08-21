"""Snapshot lifecycle.

Each build gets a versioned folder (`{strategy}_{YYYYMMDD_HHMM}_{index}`) at
`set_strategy` time. Subsequent stages write directly into that folder, so
no rename is ever required. The registry (`registry.json`) is the single
source of truth for which folder is the active build and which is the latest
sealed snapshot. The legacy `snapshots/latest` directory and rename-based
finalisation are no longer used.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path

from .paths import SnapshotPaths
from . import registry

logger = logging.getLogger(__name__)


def save_config(save_path: Path, config_path: str) -> None:
    """Copy a config file into a snapshot's configs directory.

    'save_path' must point at an existing or creatable directory.
    The source filename is preserved as the destination filename.
    """
    save_dir = Path(save_path)
    src = Path(config_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, save_dir / src.name)


def _copy_configs_into(paths: SnapshotPaths) -> None:
    """Copy each config listed in `paths.config_files` into the snapshot.
    """
    for config_path in paths.config_files:
        save_config(save_path=paths.configs, config_path=config_path)


def start_new_build(
    snapshots_dir: Path,
    registry_file: Path,
    strategy: str,
) -> str:
    """Claim a new versioned folder for a build and register it as `current`.

    Returns the versioned folder name (e.g. `dense_20260821_1430_0`).
    The folder is created, `metadata.json` is written with the strategy,
    and the registry's `current` target and `status` are updated.
    """
    snapshots_dir = Path(snapshots_dir)
    registry_file = Path(registry_file)

    name = registry.claim_snapshot_name(snapshots_dir, strategy)
    folder = snapshots_dir / name
    metadata_path = folder / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump({"strategy": strategy}, f, indent=2)

    registry.update_target(registry_file, "current", name)
    registry.update_status(registry_file, name, "building")

    return name


def finalise_snapshot(
    paths: SnapshotPaths,
    registry_file: Path,
    versioned_name: str,
) -> None:
    """Seal the snapshot in `paths.building` (a versioned folder) and mark it complete.

    No rename: the versioned folder is the same folder the build has been
    writing to. Copies configs, stamps `created_at`, makes the snapshot
    read-only, and updates the registry so `latest` resolves to this build.
    """
    registry_file = Path(registry_file)

    if not paths.building.exists():
        raise FileNotFoundError(f"Snapshot folder not found at {paths.building}")

    _copy_configs_into(paths)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(paths.metadata, "r") as f:
        metadata = json.load(f)
    metadata["created_at"] = created_at
    with open(paths.metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    registry.update_status(registry_file, versioned_name, "complete")
    registry.update_target(registry_file, "latest", versioned_name)

    # Make the snapshot read-only. Directories stay user-writable (0o755)
    # so the user can still delete the tree if needed; only files and the
    # root folder are locked down.
    for root, _, files in os.walk(paths.building):
        for f in files:
            os.chmod(Path(root) / f, 0o444)
    os.chmod(paths.building, 0o555)
