from .base_builder import BaseBuilder, _log_dataset_identity

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.sparse.tokenizer import Tokenizer
from ..retrieval.sparse.bm25indexer import BM25Indexer
from ..retrieval.chunker import chunk_recipes

from ..core.config import ENCODER_CONFIG, INDEXER_CONFIG
from ..core.snapshot.paths import SnapshotPaths
from ..core.snapshot.lifecycle import finalise_snapshot
from ..core.snapshot.chunk_sets import (
    save_chunk_sets,
    load_chunk_sets,
    save_encoded_chunk_sets,
    save_tokenized_chunk_sets,
    load_encoded_chunk_sets,
    load_tokenized_chunk_sets,
)
from ..core.snapshot.indices import (
    save_indexed_chunk_sets,
    save_bm25_index_chunk_sets,
)


class HybridBuilder(BaseBuilder):
    def chunk(self, paths: SnapshotPaths, recipes_path):
        _log_dataset_identity("chunk", paths, recipes_path)
        chunk_sets = chunk_recipes(
            strategy="hybrid",
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

        sparse = [cs for cs in chunk_sets if cs.name == "sparse"]
        dense = [cs for cs in chunk_sets if cs.name == "dense"]

        # tokenize sparse group
        tokenizer = Tokenizer()
        tokenized_chunk_sets = tokenizer.transform(sparse)

        # encode dense group
        encoder = Encoder(
            model_id=ENCODER_CONFIG["model_id"],
            batch_size=ENCODER_CONFIG["batch_size"],
            normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"],
        )
        encoded_chunk_sets = encoder.encode_chunks(dense)

        save_tokenized_chunk_sets(
            tokenized_chunk_sets=tokenized_chunk_sets,
            save_path=paths.transformed_chunks,
        )
        save_encoded_chunk_sets(
            encoded_chunk_sets=encoded_chunk_sets,
            save_path=paths.transformed_chunks,
        )

    def index(self, paths: SnapshotPaths):
        _log_dataset_identity("index", paths)
        tokenized_chunk_sets = load_tokenized_chunk_sets(
            save_path=paths.transformed_chunks,
        )
        encoded_chunk_sets = load_encoded_chunk_sets(
            save_path=paths.transformed_chunks,
        )

        # index sparse
        indexer = BM25Indexer()
        bm25_index = indexer.build(tokenized_chunk_sets[0])

        # index dense
        indexer = Indexer(
            index_type=INDEXER_CONFIG["type"],
            dimension=INDEXER_CONFIG["dimension"],
        )
        indexed_chunks_sets = indexer.index_embeddings(encoded_chunk_sets)

        save_bm25_index_chunk_sets(
            bm25_index_chunk_sets=bm25_index,
            save_path=paths.index,
        )
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
