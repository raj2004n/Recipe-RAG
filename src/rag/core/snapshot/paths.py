"""Snapshot path resolution.

Centralises every directory the snapshot subsystem touches. Callers receive a
'SnapshotPaths'.
"""

from dataclasses import dataclass
from pathlib import Path


_SNAPSHOTS_ROOT = Path("snapshots")


@dataclass(frozen=True)
class SnapshotPaths:
    """Resolved paths for a single in-progress or sealed snapshot.

    The `building` field is the snapshot's root folder. Under the new flow
    this is a versioned folder (`snapshots/{strategy}_{YYYYMMDD_HHMM}_{index}/`)
    created at the start of a build and never renamed. The `latest` symlink
    or directory is no longer a real path - the registry is the source of
    truth for "which folder is latest".

    Attributes:
        building: the snapshot's root directory.
        chunks: where chunk sets are persisted during build.
        transformed_chunks: where encoded/tokenised chunk sets are persisted.
        index: where dense / BM25 indices are persisted during build.
        configs: where copied config files live inside the snapshot.
        metadata: path to metadata.json inside the snapshot.
        registry_file: registry.json alongside snapshots/.
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

    @classmethod
    def from_paths_config(
        cls,
        paths_cfg: dict,
        versioned_name: str | None = None,
    ) -> "SnapshotPaths":
        """Resolve snapshot paths from `PATHS_CONFIG`.

        Args:
            paths_cfg: the `PATHS_CONFIG` dict, with optional `building` and
                `configs` blocks.
            versioned_name: the versioned folder name (e.g.
                `dense_20260821_1430_0`) when resolving paths for an active
                or sealed build. When given, `building` is
                `snapshots/<versioned_name>/`. When `None`, falls back to
                the legacy `snapshots/building/` for backward compatibility
                with code paths that have not yet migrated.
        """
        building_cfg = paths_cfg.get("building", {})
        configs = paths_cfg.get("configs", {})

        if versioned_name is not None:
            building_root = _SNAPSHOTS_ROOT / versioned_name
            chunks_dir = building_root / "chunks"
            transformed_chunks_dir = building_root / "transformed_chunks"
            index_dir = building_root / "index"
            configs_dir = building_root / "configs"
            metadata_dir = building_root / "metadata"
        else:
            building_root = _SNAPSHOTS_ROOT / "building"
            chunks_dir = Path(building_cfg.get("chunks", "snapshots/building/chunks"))
            transformed_chunks_dir = Path(building_cfg.get("transformed_chunks", "snapshots/building/transformed_chunks"))
            index_dir = Path(building_cfg.get("index", "snapshots/building/index"))
            configs_dir = Path(building_cfg.get("configs", "snapshots/building/configs"))
            metadata_dir = Path(building_cfg.get("metadata", "snapshots/building"))

        return cls(
            building=building_root,
            chunks=chunks_dir,
            transformed_chunks=transformed_chunks_dir,
            index=index_dir,
            configs=configs_dir,
            metadata=metadata_dir / "metadata.json",
            registry_file=_SNAPSHOTS_ROOT / "registry.json",
            config_files=tuple(configs.values()),
        )
