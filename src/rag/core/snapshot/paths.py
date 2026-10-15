"""Snapshot path resolution.

Centralises every directory the snapshot subsystem touches. Callers receive a
'SnapshotPaths'.
"""

from dataclasses import dataclass
from pathlib import Path


_SNAPSHOTS_ROOT = Path("snapshots")
_STAGING_ROOT = Path("build/staging")


@dataclass(frozen=True)
class SnapshotPaths:
    """Resolved paths for the staging tree or for a sealed snapshot.

    A build fills one fixed staging tree at `build/staging/` across its stages,
    and `finalise_build` copies that tree into a versioned folder
    (`snapshots/{strategy}_{YYYYMMDD_HHMM}_{index}/`). Resolving with a
    versioned name gives the sealed folder; resolving with no name gives the
    staging tree. `snapshots/` holds sealed, deployable folders only, and the
    registry is the source of truth for which one is newest.

    Attributes:
        building: the root directory of the staging tree or sealed snapshot.
        chunks: where chunk sets are persisted during build.
        transformed_chunks: where encoded/tokenised chunk sets are persisted.
        index: where dense / BM25 indices are persisted during build.
        configs: where copied config files live inside the snapshot.
        metadata: path to metadata.json.
        registry_file: registry.json alongside the sealed folders.
        config_files: source paths of every config file to copy into the
            snapshot, in the order they are copied.
    """
    building: Path
    chunks: Path
    transformed_chunks: Path
    index: Path
    configs: Path
    metadata: Path
    registry_file: Path
    config_files: tuple[str, ...]

    @property
    def snapshots_root(self) -> Path:
        """The directory holding the sealed snapshot folders and the registry."""
        return _SNAPSHOTS_ROOT

    @property
    def is_staging(self) -> bool:
        """True when these paths point at the staging tree rather than a sealed folder."""
        return self.building == _STAGING_ROOT

    @classmethod
    def from_paths_config(
        cls,
        paths_cfg: dict,
        versioned_name: str | None = None,
    ) -> "SnapshotPaths":
        """Resolve snapshot paths from `PATHS_CONFIG`.

        Args:
            paths_cfg: the `PATHS_CONFIG` dict, whose `configs` block lists the
                files to copy into a sealed snapshot.
            versioned_name: the versioned folder name (e.g.
                `dense_20260821_1430_0`) when resolving paths for a sealed
                build. When `None`, resolves the staging tree at
                `build/staging/`, which every build stage writes into.
        """
        configs = paths_cfg.get("configs", {})
        root = _SNAPSHOTS_ROOT / versioned_name if versioned_name is not None else _STAGING_ROOT

        return cls(
            building=root,
            chunks=root / "chunks",
            transformed_chunks=root / "transformed_chunks",
            index=root / "index",
            configs=root / "configs",
            metadata=root / "metadata.json",
            registry_file=_SNAPSHOTS_ROOT / "registry.json",
            config_files=tuple(configs.values()),
        )
