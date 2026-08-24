from .base_builder import BaseBuilder, _log_dataset_identity

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.chunker import chunk_recipes

from ..core.config import ENCODER_CONFIG, INDEXER_CONFIG
from ..core.snapshot.paths import SnapshotPaths
from ..core.snapshot.lifecycle import finalise_snapshot, sha256_of
from ..core.snapshot.chunk_sets import (
    save_chunk_sets,
    load_chunk_sets,
    save_encoded_chunk_sets,
    load_encoded_chunk_sets,
)
from ..core.snapshot.indices import save_indexed_chunk_sets


class DenseBuilder(BaseBuilder):
    def chunk(self, paths: SnapshotPaths, recipes_path):
        _log_dataset_identity("chunk", paths, recipes_path)
        chunk_sets = chunk_recipes(
            strategy="whole_recipe",
            recipes_path=recipes_path,
        )

        save_chunk_sets(
            chunk_sets=chunk_sets,
            save_path=paths.chunks,
        )

    def transform(self, paths: SnapshotPaths):
        _log_dataset_identity("transform", paths)
        chunk_sets = load_chunk_sets(
            save_path=paths.chunks,
        )

        encoder = Encoder(
            model_id=ENCODER_CONFIG["model_id"],
            batch_size=ENCODER_CONFIG["batch_size"],
            normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"],
        )
        # returns list containing one encoded chunk set
        encoded_chunk_sets = encoder.encode_chunks(chunk_sets)

        save_encoded_chunk_sets(
            encoded_chunk_sets=encoded_chunk_sets,
            save_path=paths.transformed_chunks,
        )

    def index(self, paths: SnapshotPaths):
        _log_dataset_identity("index", paths)
        encoded_chunk_sets = load_encoded_chunk_sets(
            save_path=paths.transformed_chunks,
        )

        indexer = Indexer(
            index_type=INDEXER_CONFIG["type"],
            dimension=INDEXER_CONFIG["dimension"],
        )

        indexed_chunks_sets = indexer.index_embeddings(encoded_chunk_sets)

        save_indexed_chunk_sets(
            indexed_chunk_sets=indexed_chunks_sets,
            save_path=paths.index,
        )

    def seal_snapshot(self, paths: SnapshotPaths, registry_file, versioned_name: str):
        # TODO: add checks (e.g. ensure all stages produced expected outputs).
        finalise_snapshot(
            paths=paths,
            registry_file=registry_file,
            versioned_name=versioned_name,
        )
