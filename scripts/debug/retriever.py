import argparse
from src.retrieval.indexer import Indexer 
from src.retrieval.retriever import RetrievalResult
from src.utils import flatten_config_dict, load_retriever, load_snapshot
import mlflow

""" 
- Does not handle negation well
Consider Query decomp, and knowledge graph plus more..
"""

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-retriever-experiment")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=list[str], required=False, default=["How do I make chicken curry?"])
    parser.add_argument("--top_k", type=int, required=False, default=5)
    return parser.parse_args()

def inspect_index(indexer: Indexer, chunks: list[dict]):
    # number of chunks
    print(f"Number of chunks: {len(chunks)}")
    
    # number of vectors
    print(f"Number of vectors: {indexer.index.ntotal}")

def print_results(results: list[RetrievalResult]):
    for i, r in enumerate(results):
        print(f"-"*50)
        print(f"Rank: {i}")
        print(f"Scorce: {r.score}")
        #print(f"Recipe ID: {r.recipe_id}")
        #print(f"Chunk ID: {r.chunk_id}")
        
        metadata = r.chunk["metadata"]
        print(f"Recipe Name: {metadata['recipe_name']}")

def debug_retriever(queries: list[str], top_k: int = 5):
    snapshot = load_snapshot("snapshots/latest")
    retriever = load_retriever(snapshot)
    res = {}
    for query in queries:
        res[query] = []
    #inspect_index(retriever.indexer, retriever.chunks)
    for query in queries:
        print(f"="*50)
        print(f"Query: {query}")
        print(f"="*50)

        results = retriever.retrieve(query, top_k)
        print_results(results)

        structured_results = [
            {
                "rank": i,
                "score": r.score,
                "recipe_id": r.recipe_id,
                "recipe_name": r.chunk["metadata"]["recipe_name"],
            }
            for i, r in enumerate(results)
        ]
        
        res[query] = structured_results

    with mlflow.start_run():
        mlflow.log_param("query", query)
        mlflow.log_param("top_k", top_k)
        mlflow.log_params(flatten_config_dict(snapshot.base_config))
        mlflow.log_dict(res, "results.json")

if __name__ == "__main__":
    args = parse_args()
    queries = [
        "Recipes that do not use meat",
    ]
    debug_retriever(queries, 100)