from .base_builder import BaseBuilder

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.chunker import chunk_recipes

from src.rag.core.utils import load_config
from src.rag.core.snapshot import (
    save_chunk_sets,
    load_chunk_sets,
    save_encoded_chunk_sets,
    load_encoded_chunk_sets,
    save_indexed_chunk_sets,
    finalise_snapshot
)


BASE_CONFIG = load_config("configs/base_config.yaml")
PATHS_CONFIG = load_config("configs/paths_config.yaml")

ENCODER_CONFIG = BASE_CONFIG["encoder"]
INDEXER_CONFIG = BASE_CONFIG["indexer"]
RETRIEVER_CONFIG = BASE_CONFIG["retriever"]

FILES = PATHS_CONFIG["data"]
BUILDING = PATHS_CONFIG["building"]


class DenseBuilder(BaseBuilder):
    def chunk(self):
        chunk_sets = chunk_recipes(
            strategy="whole_recipe",
            recipes_path=FILES["cleaned_recipes"]
        )
        
        save_chunk_sets(
            chunk_sets=chunk_sets,
            save_path=BUILDING["chunks"]
        )


    def transform(self):
        chunk_sets = load_chunk_sets(
            save_path=BUILDING["chunks"],
        )
        
        encoder = Encoder(
            model_id=ENCODER_CONFIG["model_id"],
            batch_size=ENCODER_CONFIG["batch_size"],
            normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"]
        )
        # returns list containing one encoded chunk set
        encoded_chunk_sets = encoder.encode_chunks(chunk_sets)

        save_encoded_chunk_sets(
            encoded_chunk_sets=encoded_chunk_sets,
            save_path=BUILDING["transformed_chunks"]
        )


    def index(self):
        encoded_chunk_sets = load_encoded_chunk_sets(
            save_path=BUILDING["transformed_chunks"]
        )

        indexer = Indexer(
            index_type=INDEXER_CONFIG["type"],
            dimension=INDEXER_CONFIG["dimension"]
        )
    
        indexed_chunks_sets = indexer.index_embeddings(encoded_chunk_sets)
    
        save_indexed_chunk_sets(
            indexed_chunk_sets=indexed_chunks_sets,
            save_path=BUILDING["index"]
        )


    def seal_snapshot(self):
        # TODO: need some checks:
        finalise_snapshot()