""" 
Script to test new dense retriever after removing accepts logic.
"""

import json 
from src.rag.core.utils import load_config
from src.rag.builder.base_builder import BaseBuilder

BASE_CONFIG = load_config("configs/base_config.yaml")
TEST_METADATA = "temp/building" # path to where temp files will be stored


def load_recipes():
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)
    return recipes

# ========= end-to-end run ============
# works
def set_strategy() -> None:
    # get strategy from config
    strategy_name = BASE_CONFIG["rag"]["strategy"]

    BaseBuilder.set_strategy(
        strategy_name=strategy_name,
        metadata_path=TEST_METADATA
    )


# works
def chunk() -> None:
    strategy = BaseBuilder.from_metadata(TEST_METADATA)
    strategy.chunk()


# works
def encode_chunk() -> None:
    strategy = BaseBuilder.from_metadata(TEST_METADATA)
    strategy.transform()

    
# works
def index_embeddings()-> None:
    strategy = BaseBuilder.from_metadata(TEST_METADATA)
    strategy.index()
    

# works
def finalise_build() -> None:
    strategy = BaseBuilder.from_metadata(TEST_METADATA)
    strategy.seal_snapshot()


# ======= other tests ========
# works
def test_from_metadata() -> None:
    "make sure the correct strategy is pulled"
    strategy = BaseBuilder.from_metadata(TEST_METADATA)
    print(strategy)


# works
def test_index_embeddings():
    "test if index embeddings correctly produces IndexedChunkSets with Chunks dict"

    from src.rag.retrieval.dense.indexer import Indexer
    from src.rag.core.snapshot import load_encoded_chunk_sets
    from itertools import islice

    BASE_CONFIG = load_config("configs/base_config.yaml")
    PATHS_CONFIG = load_config("configs/paths_config.yaml")

    TEST_PATH = PATHS_CONFIG["test"]
    INDEXER_CONFIG = BASE_CONFIG["indexer"]

    encoded_chunk_sets = load_encoded_chunk_sets(
        save_path=TEST_PATH["encoded_chunks"]
    )
    
    indexer = Indexer(
        index_type=INDEXER_CONFIG["type"],
        dimension=INDEXER_CONFIG["dimension"]
    )
    
    indexed_chunks_sets = indexer.index_embeddings(encoded_chunk_sets)
    
    chunks = indexed_chunks_sets[0].chunks

    # print out a snippet
    print(
        f"IndexedChunkSet("
        f"name={indexed_chunks_sets[0].name!r}, \n\n"
        f"index={indexed_chunks_sets[0].index!r}, \n\n"
        f"chunks={len(chunks)} items, \n\n"
        f"sample={dict(islice(chunks.items(), 3))}"
        f")"
    )


# works
def test_load_index():
    # sanity check for loading bm25 index
    
    from src.rag.core.snapshot import load_bm25_index_chunk_sets
    from pathlib import Path

    path = "temp/building"

    path_obj = Path(path)
    
    # load chunks (used to attach Chunks to IndexedChunkSets so the
    # indexer can resolve hits at query time)
    bm25_indexed_chunk_sets, bm25_chunks_by_set_name = load_bm25_index_chunk_sets(
        path_obj / "index",
        chunks_save_path=path_obj / "chunks",
    )

    print(bm25_indexed_chunk_sets[0])


if __name__ == "__main__":
    set_strategy()
    chunk()
    encode_chunk()
    index_embeddings()
    finalise_build()