"""
This is the builder base (superclass).

A builder takes the cleaned recipes as input and produces a snapshot that a
Pipeline can load for inference. This approach lets each strategy decide
which ChunkSets to feed and how to index them, without the retriever
components having to know about chunk-set filtering.

4 steps:
    chunk         - split documents into ChunkSets
    transform     - encode/tokenise ChunkSets into index ready form
    index         - build FAISS / BM25 indices over the transformed chunks
    seal_snapshot - stamp created_at, copy configs, mark the build complete

Subclasses (DenseBuilder, HybridBuilder) decide which ChunkSets to feed and
what to encode vs tokenise. The active build folder is passed in as a
SnapshotPaths instance; the builder does not read BUILDING_PATHS or any
module-level path singleton.
"""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path

from ..core.checksum import file_checksum
from ..core.snapshot.paths import SnapshotPaths
from ..core.snapshot import lifecycle

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
        snapshots_dir,
        registry_file,
    ) -> str:
        """Resolve strategy from `base_config` and start a new versioned build.

        Returns the versioned folder name (e.g. `dense_20260821_1430_0`).
        """
        strategy_name = base_config["rag"]["strategy"]
        return cls._set_strategy(
            strategy_name=strategy_name,
            snapshots_dir=snapshots_dir,
            registry_file=registry_file,
        )

    @classmethod
    def _set_strategy(
        cls,
        strategy_name: str,
        snapshots_dir,
        registry_file,
    ) -> str:
        """Start a new build with the given strategy.

        Delegates to `lifecycle.start_new_build`, which creates the versioned
        folder, writes `metadata.json`, and registers the build as
        `current` / `building` in the registry.
        """
        if strategy_name not in {"dense", "hybrid"}:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return lifecycle.start_new_build(
            snapshots_dir=snapshots_dir,
            registry_file=registry_file,
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
    def seal_snapshot(self, paths: SnapshotPaths, registry_file, versioned_name: str):
        """Seal the in-progress snapshot so Pipeline can load it.

        Copies configs into the snapshot, stamps `created_at`, marks the
        build complete in the registry, and points the `latest` target at
        this build.
        """
