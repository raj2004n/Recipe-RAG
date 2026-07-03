import yaml
import json
import argparse
import time
from src.models.retriever import Retriever
from src.models.encoder import Encoder
from src.models.indexer import Indexer 
from src.models.retriever import RetrievalResult

# TODO: Measure time taken to load, encode, search, and save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False, default="How do I make chicken curry?")
    parser.add_argument("--top_k", type=int, required=False, default=5)
    return parser.parse_args()

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

    
    start = time.time()
    chunks = load_chunks()
    end = time.time()
    chunks_time = end - start

    encoder = Encoder(config) # uses pretrained model, hence no load
    
    start2 = time.time()
    indexer = load_indexer()
    end2 = time.time()
    index_time = end - start

    retriever = Retriever(
        encoder=encoder,
        indexer=indexer,
        chunks=chunks,    
        config=config
    )
    end3 = time.time()
    retriever_time = end3 - end2

    return retriever, chunks_time, index_time, retriever_time

def inspect_index(indexer: Indexer, chunks: list[dict]):
    # number of chunks
    print(f"Number of chunks: {len(chunks)}")
    
    # number of vectors
    print(f"Number of vectors: {indexer.index.ntotal}")

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
    retriever, chunks_time, index_time, retriever_time = load_retriever(config)
    print(f"="*50)
    print(f"Time taken to load chunks: {chunks_time}")
    print(f"Time taken to load index: {index_time}")
    print(f"Total time taken to load retriever: {retriever_time}")
    print(f"="*50)

    inspect_index(retriever.indexer, retriever.chunks)

    print(f"="*50)
    print(f"Query: {query}")
    print(f"="*50)
    
    results = retriever.retrieve(query, top_k)
    print_results(results)

if __name__ == "__main__":
    args = parse_args()
    debug_retriever(args.query, args.top_k)