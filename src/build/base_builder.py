from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path

from ..core.checksum import file_checksum
from ..snapshot.paths import SnapshotPaths
from ..snapshot import lifecycle

logger = logging.getLogger(__name__)


def _log_dataset_identity(stage: str, paths: SnapshotPaths, recipes_path=None) -> None:
    """Emit a single log line that pins the snapshot to its dataset.

    At the chunk stage we pass the recipes_path explicitly so the log shows
    the path being read right now. At later stages we re-read metadata.json
    instead, because what matters is the identity captured at set_strategy -
    not the file's current state. This is what makes a mid-build preprocessing
    swap visible in the log trail.
    """
    metadata = json.loads(paths.metadata.read_text())
    key = metadata.get("dataset_key", "<unset>")

    if recipes_path is not None:
        try:
            checksum = file_checksum(Path(recipes_path))
        except OSError as e:
            checksum = f"<unreadable: {e}>"
        logger.info(
            "%s: dataset_key=%s recipes_path=%s checksum=%s",
            stage, key, recipes_path, checksum,
        )
    else:
        logger.info("%s: dataset_key=%s", stage, key)


class BaseBuilder(ABC):
    """Base class for build-time orchestration of a retrieval index."""

    @classmethod
    def set_strategy_from_config(
        cls,
        base_config: dict,
        staging_dir,
    ) -> None:
        """Resolve strategy from `base_config` and open the staging tree for a build.
        """
        strategy_name = base_config["rag"]["strategy"]
        return cls._set_strategy(
            strategy_name=strategy_name,
            staging_dir=staging_dir,
        )

    @classmethod
    def _set_strategy(
        cls,
        strategy_name: str,
        staging_dir,
    ) -> None:
        """Open the staging tree for a build with the given strategy.

        Validates the strategy and delegates to `lifecycle.start_new_build`,
        which empties the staging tree, writes `metadata.json`, and claims the
        tree as in progress.
        """
        if strategy_name not in {"dense", "hybrid"}:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return lifecycle.start_new_build(
            staging_dir=staging_dir,
            strategy=strategy_name,
        )

    @classmethod
    def from_strategy_name(cls, strategy_name: str) -> "BaseBuilder":
        """Return a Builder subclass for the given strategy name.
        """
        from .dense_builder import DenseBuilder
        from .hybrid_builder import HybridBuilder

        builders = {"dense": DenseBuilder, "hybrid": HybridBuilder}
        if strategy_name not in builders:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return builders[strategy_name]()

    @classmethod
    def from_paths(cls, paths: SnapshotPaths) -> "BaseBuilder":
        """Read `metadata.json` from `paths.building` and return the matching Builder.
        """
        import json

        with open(paths.metadata, "r") as f:
            metadata = json.load(f)
        return cls.from_strategy_name(metadata["strategy"])

    @abstractmethod
    def chunk(self, paths: SnapshotPaths):
        """Split source documents into ChunkSets and persist them."""

    @abstractmethod
    def transform(self, paths: SnapshotPaths):
        """Encode/tokenise ChunkSets into the form the indexers consume."""

    @abstractmethod
    def index(self, paths: SnapshotPaths):
        """Build FAISS and/or BM25 indices over the transformed chunks."""

    @abstractmethod
    def seal_snapshot(self, paths: SnapshotPaths, registry_file) -> str:
        """Seal the staged build into a new versioned snapshot a Pipeline can load.

        Claims a versioned folder, copies the staging tree into it, copies the
        configs, stamps `created_at`, marks the build complete in the registry,
        and points `latest_build` at it. Returns the versioned folder name.
        """
