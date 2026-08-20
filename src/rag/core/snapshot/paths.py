"""Snapshot path resolution.

Centralises every directory the snapshot subsystem touches. Callers receive a
'SnapshotPaths'.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnapshotPaths:
    """Resolved paths for the snapshot subsystem.

    Attributes:
        building: the in-progress snapshot directory (`snapshots/building`).
        latest:  the live snapshot directory (`snapshots/latest`).
        chunks:  where chunk sets are persisted during build.
        index:   where dense / BM25 indices are persisted during build.
        configs: where copied config files live inside the snapshot.
        metadata: path to metadata.json inside the snapshot.
        config_files: source paths of every config file to copy into the
            snapshot, in the order they are copied.
    """
    building: Path
    latest: Path
    chunks: Path
    index: Path
    configs: Path
    metadata: Path
    config_files: tuple[str, ...]

    @classmethod
    def from_paths_config(cls, paths_cfg: dict) -> "SnapshotPaths":
        """Resolve snapshot paths from `PATHS_CONFIG`.

        Args:
            paths_cfg: the full `paths_config.yaml` dict. Reads
                `paths_cfg["building"]` for snapshot directories and
                `paths_cfg["configs"]` for the list of config files to copy
                into the sealed snapshot.
        """
        building = paths_cfg["building"]
        configs = paths_cfg["configs"]

        building_root = Path("snapshots/building")
        return cls(
            building=building_root,
            latest=building_root.parent / "latest",
            chunks=Path(building["chunks"]),
            index=Path(building["index"]),
            configs=Path(building["configs"]),
            metadata=Path(building["metadata"]) / "metadata.json",
            config_files=tuple(configs.values()),
        )
