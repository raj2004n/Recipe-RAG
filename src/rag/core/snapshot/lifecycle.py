"""Snapshot lifecycle.

A build fills one fixed staging tree at `build/staging/` across its stages,
and `finalise_build` copies that tree into a versioned folder
(`{strategy}_{YYYYMMDD_HHMM}_{index}`) under `snapshots/`. Copying rather than
renaming keeps the declared staging outputs in place, so DVC can skip the
stages whose inputs and outputs are unchanged, and it leaves the versioned
name unclaimed until a build has actually produced something. The registry
(`registry.json`) is the single source of truth for the newest sealed build.

A build pins its dataset identity at `set_strategy` (path + checksum, plus the
registry key) so a concurrent preprocess cannot silently desync the indexed
model from the metadata. The advisory lock in `recipe_file_lock` exists to
make any such swap visible in the log; the real protection is that the
identity is captured at the start of the build.
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


# The claim marks a build as in progress in the staging tree, and the lock
# keeps two processes from writing that tree at once. DVC runs each stage as
# its own process, so no single process spans a whole build; the claim is what
# spans it, and the lock covers the span of one stage.
_CLAIM_NAME = ".claim"
_LOCK_NAME = ".lock"
_STAGE_SUBDIRS = ("chunks", "transformed_chunks", "index")


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


@contextlib.contextmanager
def staging_lock(staging_dir: Path) -> Iterator[None]:
    """Hold an advisory lock on the staging tree for the span of one stage.

    This stops a second process from writing the tree while a stage is
    running, for example a stray `--run chunk` launched while a build is
    under way. It is taken per stage rather than per build because DVC runs
    each stage as its own process, so nothing longer-lived is available to
    hold it. Whole-build exclusivity comes from the claim instead, which
    `start_new_build` refuses to write twice.
    """
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(staging_dir / _LOCK_NAME), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(
                f"another process is writing {staging_dir}; refusing to run a stage"
            ) from e
        logger.info("acquired staging lock on %s", staging_dir)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            logger.info("released staging lock on %s", staging_dir)
    finally:
        os.close(fd)


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


def read_claim(staging_dir: Path) -> Optional[dict]:
    """Return the claim left by an in-progress build, or None when there is none."""
    claim_path = Path(staging_dir) / _CLAIM_NAME
    try:
        return json.loads(claim_path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def start_new_build(
    staging_dir: Path,
    strategy: str,
    dataset_key: Optional[str] = None,
) -> None:
    """Open the staging tree for a build and claim it as in progress.

    Empties the staging subdirectories so the build starts from a clean tree,
    writes `metadata.json` with the strategy and (when given) the dataset key,
    and writes the claim that marks a build as under way. A second build is
    refused while that claim exists, since two builds writing one staging tree
    would interleave their outputs.

    Capturing the dataset key here rather than at finalise is what protects
    the build from a mid-flight preprocessing swap: by the time the build
    finishes, the key and the indexed model agree, because both were pinned
    against the same dataset.
    """
    staging_dir = Path(staging_dir)
    if read_claim(staging_dir) is not None:
        raise RuntimeError(
            f"a build is already in progress in {staging_dir}; "
            "finalise it or clear the staging claim before starting another"
        )

    for subdir in _STAGE_SUBDIRS:
        target = staging_dir / subdir
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

    metadata = {"strategy": strategy}
    if dataset_key is not None:
        metadata["dataset_key"] = dataset_key
    (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    claim = {"strategy": strategy, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (staging_dir / _CLAIM_NAME).write_text(json.dumps(claim, indent=2))
    logger.info("claimed staging tree %s for a %s build", staging_dir, strategy)


def finalise_snapshot(
    staging_paths: SnapshotPaths,
    registry_file: Path,
) -> str:
    """Copy the staging tree into a newly claimed sealed folder and mark it complete.

    Claims a fresh versioned name, copies the staging tree into it, copies the
    configs, stamps `created_at`, makes the folder read-only, and points the
    registry's `latest_build` at it. The staging tree keeps its bytes, so the
    DVC outputs declared for the stage stay in place, and the claim is cleared
    so the next build can open the tree again.

    Returns the versioned folder name (e.g. `dense_20260821_1430_0`).
    """
    registry_file = Path(registry_file)
    staging_dir = Path(staging_paths.building)

    if not staging_paths.metadata.exists():
        raise FileNotFoundError(
            f"No build in progress at {staging_dir}: metadata.json is missing. "
            "Run 'set_strategy' first."
        )

    with open(staging_paths.metadata, "r") as f:
        metadata = json.load(f)

    name = registry.claim_snapshot_name(staging_paths.snapshots_root, metadata["strategy"])
    sealed_dir = staging_paths.snapshots_root / name
    # The claim already created the folder, so the copy writes into it rather
    # than expecting to create it.
    shutil.copytree(
        staging_dir,
        sealed_dir,
        ignore=shutil.ignore_patterns(_CLAIM_NAME, _LOCK_NAME),
        dirs_exist_ok=True,
    )

    (sealed_dir / "configs").mkdir(parents=True, exist_ok=True)
    for config_path in staging_paths.config_files:
        save_config(save_path=sealed_dir / "configs", config_path=config_path)

    sealed_metadata_path = sealed_dir / "metadata.json"
    with open(sealed_metadata_path, "r") as f:
        sealed_metadata = json.load(f)
    sealed_metadata["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(sealed_metadata_path, "w") as f:
        json.dump(sealed_metadata, f, indent=2)

    registry.update_status(registry_file, name, "complete")
    registry.update_target(registry_file, "latest_build", name)

    # Make the sealed snapshot read-only. Directories stay user-writable
    # (0o755) so the user can still delete the tree if needed; only files and
    # the root folder are locked down. The staging tree is untouched, so the
    # next build can write into it.
    for root, _, files in os.walk(sealed_dir):
        for f in files:
            os.chmod(Path(root) / f, 0o444)
    os.chmod(sealed_dir, 0o555)

    with contextlib.suppress(FileNotFoundError):
        (staging_dir / _CLAIM_NAME).unlink()

    return name
