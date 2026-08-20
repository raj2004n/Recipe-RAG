from .base_builder import BaseBuilder

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.sparse.tokenizer import Tokenizer
from ..retrieval.sparse.bm25indexer import BM25Indexer
from ..retrieval.chunker import chunk_recipes

from src.rag.core.config import (
    ENCODER_CONFIG, INDEXER_CONFIG,
    DATA_PATHS, BUILDING_PATHS, PATHS_CONFIG,
)
from src.rag.core.snapshot.chunk_sets import (
    save_chunk_sets,
    load_chunk_sets,
    save_encoded_chunk_sets,
    save_tokenized_chunk_sets,
    load_encoded_chunk_sets,
    load_tokenized_chunk_sets,
)
from src.rag.core.snapshot.indices import (
    save_indexed_chunk_sets,
    save_bm25_index_chunk_sets,
)
from src.rag.core.snapshot.lifecycle import finalise_snapshot
from src.rag.core.snapshot.paths import SnapshotPaths


SNAPSHOT_PATHS = SnapshotPaths.from_paths_config(PATHS_CONFIG)


class HybridBuilder(BaseBuilder):
    def chunk(self):
        chunk_sets = chunk_recipes(
            strategy="hybrid",
            recipes_path=DATA_PATHS["cleaned_recipes"]
        )

        save_chunk_sets(
            chunk_sets=chunk_sets,
            save_path=BUILDING_PATHS["chunks"]
        )


    def transform(self):
        chunk_sets = load_chunk_sets(
            save_path=BUILDING_PATHS["chunks"],
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
            normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"]
        )
        encoded_chunk_sets = encoder.encode_chunks(dense)

        save_tokenized_chunk_sets(
            tokenized_chunk_sets=tokenized_chunk_sets,
            save_path=BUILDING_PATHS["transformed_chunks"]
        )
        save_encoded_chunk_sets(
            encoded_chunk_sets=encoded_chunk_sets,
            save_path=BUILDING_PATHS["transformed_chunks"]
        )


    def index(self):
        tokenized_chunk_sets = load_tokenized_chunk_sets(
            save_path=BUILDING_PATHS["transformed_chunks"]
        )
        encoded_chunk_sets = load_encoded_chunk_sets(
            save_path=BUILDING_PATHS["transformed_chunks"]
        )

        # index sparse
        indexer = BM25Indexer()
        bm25_index = indexer.build(tokenized_chunk_sets[0])

        # index dense
        indexer = Indexer(
            index_type=INDEXER_CONFIG["type"],
            dimension=INDEXER_CONFIG["dimension"]
        )
        indexed_chunks_sets = indexer.index_embeddings(encoded_chunk_sets)

        save_bm25_index_chunk_sets(
            bm25_index_chunk_sets=bm25_index,
            save_path=BUILDING_PATHS["index"]
        )
        save_indexed_chunk_sets(
            indexed_chunk_sets=indexed_chunks_sets,
            save_path=BUILDING_PATHS["index"]
        )


    def seal_snapshot(self):
        # TODO: need some checks:
        finalise_snapshot(SNAPSHOT_PATHS)