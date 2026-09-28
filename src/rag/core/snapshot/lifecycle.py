"""Snapshot lifecycle.

Each build gets a versioned folder (`{strategy}_{YYYYMMDD_HHMM}_{index}`) at
`set_strategy` time. Subsequent stages write directly into that folder, so
no rename is ever required. The registry (`registry.json`) is the single
source of truth for which folder is the active build and which is the latest
sealed snapshot. The legacy `snapshots/latest` directory and rename-based
finalisation are no longer used.

A build pins its dataset identity at `set_strategy` (path + checksum, plus
the registry key) so a concurrent preprocess cannot silently desync the
indexed model from the metadata. The advisory lock in `recipe_file_lock`
exists to make any such swap visible in the log; the real protection is that
the identity is captured at the start of the build.
"""

import contextlib
import fcntl
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional

from .paths import SnapshotPaths
from . import registry

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def recipe_file_lock(recipes_path: Path) -> Iterator[None]:
    """Hold an advisory OS-level lock on `recipes_path` for the build's duration.

    Advisory only: a process that ignores the lock will not be blocked, but
    `scripts/preprocess` and any future tooling that respects it will refuse
    to overwrite the file while a build is in flight. The lock is released
    on context exit, including exceptions.

    The file is opened in read mode so the lock survives even when the
    caller only needs the recipe identity (no write happens here). On
    POSIX this is `fcntl.flock`; the project is Linux-only.
    """
    path = Path(recipes_path)
    lock_path = path.with_suffix(path.suffix + ".buildlock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        logger.info(
            "acquired advisory build lock on %s (via %s)",
            path,
            lock_path,
        )
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            logger.info("released advisory build lock on %s", path)
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


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
    dataset_key: Optional[str] = None,
) -> str:
    """Claim a new versioned folder for a build and register it as `build_in_progress`.

    Returns the versioned folder name (e.g. `dense_20260821_1430_0`).
    The folder is created, `metadata.json` is written with the strategy and
    (optionally) the dataset key, and the registry's `build_in_progress` target
    and `status` are updated. Capturing the dataset key at `set_strategy` time
    rather than at finalise is what protects the build from a mid-flight
    preprocessing swap: by the time the build finishes, the key and the
    indexed model agree, because both were pinned against the same dataset.
    """
    snapshots_dir = Path(snapshots_dir)
    registry_file = Path(registry_file)

    name = registry.claim_snapshot_name(snapshots_dir, strategy)
    folder = snapshots_dir / name
    metadata_path = folder / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"strategy": strategy}
    if dataset_key is not None:
        metadata["dataset_key"] = dataset_key
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # The status is written before the target, because a target that names a
    # folder which is not `building` is dropped on read. The other order would
    # write the claim while the folder had no status yet, and the next step's
    # read would drop the claim it had just made.
    registry.update_status(registry_file, name, "building")
    registry.update_target(registry_file, "build_in_progress", name)

    return name


def finalise_snapshot(
    paths: SnapshotPaths,
    registry_file: Path,
    versioned_name: str,
) -> None:
    """Seal the snapshot in `paths.building` (a versioned folder) and mark it complete.

    No rename: the versioned folder is the same folder the build has been
    writing to. Copies configs, stamps `created_at`, makes the snapshot
    read-only, and updates the registry so `latest_build` resolves to this
    build and `build_in_progress` stops naming it, since the build it named is
    no longer in progress.
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
    registry.update_target(registry_file, "latest_build", versioned_name)
    registry.clear_target(registry_file, "build_in_progress", versioned_name)

    # Make the snapshot read-only. Directories stay user-writable (0o755)
    # so the user can still delete the tree if needed; only files and the
    # root folder are locked down.
    for root, _, files in os.walk(paths.building):
        for f in files:
            os.chmod(Path(root) / f, 0o444)
    os.chmod(paths.building, 0o555)
