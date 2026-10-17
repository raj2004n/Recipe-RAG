from __future__ import annotations

import time
from pathlib import Path

from ...core.checksum import file_checksum
from .. import registry


def _derive_key(path: Path, checksum: str) -> str:
    """Build a stable key from the file's name and a checksum prefix.

    Format: `<stem>_<checksum-prefix>_<YYYYMMDD>`. Stable across re-registers
    of the same file because both stem and checksum are stable. Different
    files at the same path get different keys because the checksum differs.
    """
    stem = path.stem
    prefix = checksum.split(":", 1)[-1][:8]
    date = time.strftime("%Y%m%d")
    return f"{stem}_{prefix}_{date}"


def register_dataset(
    registry_file: Path,
    path: str | Path,
    name: str | None = None,
) -> str:
    """Register a dataset and return its key.

    Idempotent: if the same file (by checksum) is already registered under
    some key, that key is returned and the registry is left unchanged. If
    the file at `path` has a different checksum from any existing entry,
    a new key is generated (auto-derived, or `name` if the caller pinned
    one).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot register dataset: {path} is not a file")

    checksum = file_checksum(path)
    size = path.stat().st_size

    existing = registry.list_datasets(registry_file)
    for key, entry in existing.items():
        if entry.get("checksum") == checksum and entry.get("path") == str(path):
            return key

    key = name or _derive_key(path, checksum)
    entry = registry.DatasetEntry(
        path=str(path),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        checksum=checksum,
        size_bytes=size,
    )
    registry.upsert_dataset(registry_file, key, entry)
    return key


def get_dataset(registry_file: Path, key: str):
    """Thin wrapper around `registry.get_dataset` so callers do not need
    to import the registry module directly."""
    return registry.get_dataset(registry_file, key)
