from .base import BaseStrategy

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.sparse.tokenizer import Tokenizer
from ..retrieval.sparse.bm25indexer import BM25Indexer
from ..retrieval.chunker import chunk_recipes

from src.rag.core.utils import load_config
from src.rag.core.snapshot import (
    save_chunk_sets,
    load_chunk_sets,
    save_encoded_chunk_sets,
    save_tokenized_chunk_sets,
    load_encoded_chunk_sets,
    load_tokenized_chunk_sets,
    save_indexed_chunk_sets,
    save_bm25_index_chunk_sets,
    finalise_snapshot
)

BASE_CONFIG = load_config("configs/base_config.yaml")
PROMPTS_CONFIG = load_config("configs/prompts_config.yaml")
PATHS_CONFIG = load_config("configs/paths_config.yaml")

FILES = PATHS_CONFIG["files"]
DIRECTORIES = PATHS_CONFIG["directories"]

TEST_PATH = PATHS_CONFIG["test"]
ENCODER_CONFIG = BASE_CONFIG["encoder"]
INDEXER_CONFIG = BASE_CONFIG["indexer"]
RETRIEVER_CONFIG = BASE_CONFIG["retriever"]


class HybridStrategy(BaseStrategy):
    def chunk(self):
        chunk_sets = chunk_recipes(
            strategy="hybrid",
            recipes_path=FILES["cleaned_recipes"]
        )

        save_chunk_sets(
                    chunk_sets=chunk_sets,
                    save_path=TEST_PATH["chunks"]
                )


    def transform(self):
        chunk_sets = load_chunk_sets(
            save_path=TEST_PATH["chunks"],
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
            save_path=TEST_PATH["tokenized_chunks"]
        )
        save_encoded_chunk_sets(
            encoded_chunk_sets=encoded_chunk_sets,
            save_path=TEST_PATH["encoded_chunks"]
        )

    
    def index(self):
        tokenized_chunk_sets = load_tokenized_chunk_sets(
            save_path=TEST_PATH["tokenized_chunks"]
        )
        encoded_chunk_sets = load_encoded_chunk_sets(
            save_path=TEST_PATH["encoded_chunks"]
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
            save_path=TEST_PATH["index"]
        )
        save_indexed_chunk_sets(
            indexed_chunk_sets=indexed_chunks_sets,
            save_path=TEST_PATH["index"]
        )


    def finalise_build(self):
        # TODO: need some checks:
        finalise_snapshot()