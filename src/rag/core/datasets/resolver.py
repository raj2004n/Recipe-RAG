from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..checksum import file_checksum


class MissingDatasetError(LookupError):
    """The dataset key is not registered, or the registered file is gone."""


class ChecksumMismatchError(RuntimeError):
    """The file at the registered path does not match its registered checksum."""


@dataclass(frozen=True)
class ResolvedDataset:
    """The result of a successful resolve. `path` is openable from the service."""

    key: str
    path: Path
    checksum: str
    source: str   # "local" | "s3" | "gcs" - included so /info can report provenance


class DatasetResolver(Protocol):
    """Resolve a dataset key to a ResolvedDataset. Cloud impls slot in later."""

    def resolve(self, key: str) -> ResolvedDataset: ...
    def exists(self, key: str) -> bool: ...


class LocalDatasetResolver:
    """Resolve datasets registered with a project-relative or absolute `path`.

    The resolver reads the dataset entry from the snapshot registry, then
    verifies the file exists at the registered path and that its checksum
    still matches. Failures are surfaced as exceptions; the lifespan turns
    them into exit 2.
    """

    def __init__(self, registry_path: Path, registry_get_dataset):
        """`registry_get_dataset` is `registry.get_dataset` injected so this
        module does not import the registry module at import time (which
        would create a circular dependency through `snapshot.registry`)."""
        self._registry_path = Path(registry_path)
        self._get_dataset = registry_get_dataset

    def resolve(self, key: str) -> ResolvedDataset:
        entry = self._get_dataset(self._registry_path, key)
        if entry is None:
            raise MissingDatasetError(
                f"Dataset {key!r} is not registered. "
                f"Run the build pipeline to register it before loading this snapshot."
            )

        path_str = entry.get("path")
        if not path_str:
            raise MissingDatasetError(
                f"Dataset {key!r} has no `path` set. Cloud-backed datasets "
                f"need a resolver that knows how to fetch them."
            )

        path = Path(path_str)
        if not path.is_file():
            raise MissingDatasetError(
                f"Dataset {key!r} is registered at {path} but the file is "
                f"missing. Refusing to start with a stale registry entry."
            )

        registered_checksum = entry.get("checksum", "")
        actual_checksum = file_checksum(path)
        if registered_checksum and actual_checksum != registered_checksum:
            raise ChecksumMismatchError(
                f"Dataset {key!r} at {path} has checksum {actual_checksum} "
                f"but registry says {registered_checksum}. Refusing to start "
                f"to prevent querying against a stale dataset."
            )

        return ResolvedDataset(
            key=key,
            path=path,
            checksum=actual_checksum,
            source="local",
        )

    def exists(self, key: str) -> bool:
        try:
            self.resolve(key)
        except (MissingDatasetError, ChecksumMismatchError):
            return False
        return True
