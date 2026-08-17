""" 
Script to test hybrid strategy.
"""

import json 
from src.rag.core.utils import load_config
from src.rag.core.snapshot import load_chunk_sets
from src.rag.strategy.base import BaseStrategy

BASE_CONFIG = load_config("configs/base_config.yaml")
PROMPTS_CONFIG = load_config("configs/prompts_config.yaml")
PATHS_CONFIG = load_config("configs/paths_config.yaml")

FILES = PATHS_CONFIG["files"]
DIRECTORIES = PATHS_CONFIG["directories"]

ENCODER_CONFIG = BASE_CONFIG["encoder"]
INDEXER_CONFIG = BASE_CONFIG["indexer"]
RETRIEVER_CONFIG = BASE_CONFIG["retriever"]

TEST_METADATA = "temp/building" # path to where temp files will be stored


def load_recipes():
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)
    return recipes


# works
def set_strategy() -> None:
    # get strategy from config
    strategy_name = BASE_CONFIG["rag"]["strategy"]

    BaseStrategy.set_strategy(
        strategy_name=strategy_name,
        metadata_path=TEST_METADATA
    )


# works
def chunk() -> None:
    strategy = BaseStrategy.from_metadata(TEST_METADATA)
    strategy.chunk()


# works
def encode_chunk() -> None:
    strategy = BaseStrategy.from_metadata(TEST_METADATA)
    strategy.transform()


# works 
def index_embeddings()-> None:
    strategy = BaseStrategy.from_metadata(TEST_METADATA)
    strategy.index()
    

# works
def finalise_strategy() -> None:
    strategy = BaseStrategy.from_metadata(TEST_METADATA)
    strategy.finalise_build()


# ====== other tests =========
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
    #set_strategy()
    #chunk()
    #encode_chunk()
    #index_embeddings()
    finalise_strategy()

    #test_load_index()