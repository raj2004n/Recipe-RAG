import argparse
from src.retrieval.indexer import Indexer 
from src.retrieval.retriever import RetrievalResult
from src.utils import load_config, load_retriever

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False, default="How do I make chicken curry?")
    parser.add_argument("--top_k", type=int, required=False, default=5)
    return parser.parse_args()

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
    # additionally returns time taken to load chunks, index, retriever
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