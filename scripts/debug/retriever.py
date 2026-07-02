import yaml
import json

from src.models.retriever import Retriever
from src.models.encoder import Encoder
from src.models.indexer import Indexer 
from src.models.retriever import RetrievalResult

def load_config(config_file: str = "configs/base_config.yaml") -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_chunks(path: str = "models/retriever/chunks.json") -> list[dict]:
    with open(path) as f:
        chunks = json.load(f)
    return chunks

def load_indexer(path: str = "models/retriever/index.faiss") -> Indexer:
    indexer = Indexer(1) # placeholder, overwrite with load
    indexer.load(path)
    return indexer

def load_retriever(config: dict) -> Retriever:
    encoder = Encoder(config)
    indexer = load_indexer()
    chunks = load_chunks()

    retriever = Retriever(
        encoder=encoder,
        indexer=indexer,
        chunks=chunks,    
        config=config
    )

    return retriever

def insepect_index(indexer, chunks):
    pass

def print_results(results: list[RetrievalResult]):
    for i, r in enumerate(results):
        print(f"Rank: {i}")
        print(f"Scorce: {r.score}")
        print(f"Recipe ID: {r.recipe_id}")
        print(f"Chunk ID: {r.chunk_id}")
        
        metadata = r.chunk["metadata"]
        print(f"Recipe Name: {metadata['recipe_name']}")
        print(f"Recipe ID: {metadata['recipe_id']}")

def debug_retriever(query: str, top_k: int = 5):
    config = load_config()
    retriever = load_retriever(config)

    print(f"Query: {query}")

    results = retriever.retrieve(query, top_k)
    print_results(results)

if __name__ == "__main__":
    debug_retriever("How do I make chicken curry?")