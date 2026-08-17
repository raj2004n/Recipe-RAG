"""
This is the builder base (superclass). 

A builder builds takes the cleaned recipes as input and outputs a snapshot
that a Pipeline can load for inference.

This approach allows me to effectively hard-code a strategy / framework
for different RAG architectures I want to try out, without adding complexity
to the retriever components by incorporating filtering, etc, logic. 

Another reason I did it this is so the build_rag script only needs to speak
to the builder for building the retriever components. Thus, simpler build script,
and simple dvc stages.

4 steps:
    chunk         - split documents into ChunkSets
    transform     - encode/tokenise ChunkSets into index ready form
    index         - build FAISS / BM25 indices over the transformed chunks
    seal_snapshot - seal the snapshot (add id, etc into metadata, ..), ready to use for Pipeline 

As mentioned before, the subclasses (DenseBuilder, HybridBuilder) decide what 
ChunkSets to feed, what Chunks to encode / tokenize etc.
"""

from abc import ABC, abstractmethod
import json
import os


class BaseBuilder(ABC):
    """Base class for build-time orchestration of a retrieval index."""

    @classmethod
    def set_strategy(cls, strategy_name: str, metadata_path: str) -> None:
        """Persist the chosen retrieval strategy to the build's metadata.

        Writes `{metadata_path}/metadata.json` containing a single
        `strategy` field. This file is later read by `from_metadata` and
        by `load_snapshot` to decide which Pipeline to instantiate.

        Args:
            strategy_name: one of "dense" or "hybrid".
            metadata_path: directory in which to write metadata.json.
                           Created if it does not exist.
        """
        # import inside the method to avoid circular imports when dense inherits base
        # downside, builders dict duplicated in set_strategy, from_metadata
        from .dense import DenseBuilder
        from .hybrid import HybridBuilder

        builders = {
            "dense": DenseBuilder,
            "hybrid": HybridBuilder
        }

        if strategy_name not in builders:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # save strategy into metadata
        if metadata_path is None:
            os.makedirs(metadata_path, exist_ok=True)
            metadata_path = os.path.join("snapshots/building", "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "strategy": strategy
                }, f, indent=2)
        else:
            os.makedirs(metadata_path, exist_ok=True)
            metadata_path = os.path.join(metadata_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "strategy": strategy_name
                }, f, indent=2)


    @classmethod
    def from_metadata(cls, metadata_path: str) -> "BaseBuilder":
        """Retrieve retrieval strategy.

        Reads `{metadata_path}/metadata.json` and returns the matching
        Builder subclass instance. Used by the per-step debug scripts
        so each stage can re-enter the same build without re-running the
        earlier steps.
        """
        # import inside the method to avoid circular imports when dense inherits base
        # downside, builders dict duplicated in set_strategy, from_metadata
        from .dense import DenseBuilder
        from .hybrid import HybridBuilder

        builders = {
            "dense": DenseBuilder,
            "hybrid": HybridBuilder
        }

        metadata_path = os.path.join(metadata_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        strategy_name = metadata["strategy"]

        if strategy_name not in builders:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return builders[strategy_name]()


    @abstractmethod
    def chunk(self):
        """Split source documents into ChunkSets and persist them."""


    @abstractmethod
    def transform(self):
        """Encode/tokenise ChunkSets into the form the indexers consume."""


    @abstractmethod
    def index(self):
        """Build FAISS and/or BM25 indices over the transformed chunks."""


    @abstractmethod
    def seal_snapshot(self):
        """Seal the in-progress snapshot so Pipeline.load_snapshot can read it.

        Renames the building directory to latest, copies configs into
        the snapshot, and stamps metadata.json with id + created_at.
        After this returns the snapshot must not be written to again.
        """