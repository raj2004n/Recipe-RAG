import json
import mlflow
import torch
import pandas as pd
from src.utils import load_retriever
from .rerank import recipe_to_string 
from src.snapshot import load_snapshot
from scripts.debug.prompt_decomp import decompose_query
from src.retrieval.indexer import Indexer
from sentence_transformers import CrossEncoder

# retriever score is too low still, even after chunking..

mlflow.set_tracking_uri("http://localhost:5000")

def inspect_index(indexer: Indexer, chunks: list[dict]):
    # number of chunks
    print(f"Number of chunks: {len(chunks)}")
    
    # number of vectors
    print(f"Number of vectors: {indexer.index.ntotal}")

def test_new_retriever(query: str):
    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)

    # load snapshot, retriever
    snapshot = load_snapshot("snapshots/latest")
    retriever = load_retriever(snapshot)

    results = retriever.retrieve(query)
    
    recipe_ids = [r.chunk.recipe_id for r in results]
    chunk_types = [r.chunk.chunk_type for r in results]
    texts = [r.chunk.text for r in results]

    retrieved_recipes = [recipes[recipe_id]["recipe_name"] for recipe_id in recipe_ids]

    print(retrieved_recipes)

def test_query_decomp():
    #TODO: retrieve non decomposed query? Only retrieving inc and exc, is fine, but really relies on llm
    # Dense retrievers are helpful for complex look ups. Current llm simplies too much.
    # hybrid approach is viable, but there would be too many recipes with e.g. chicken
    # can try to extract negation from user query, and preserve the query otherwise.
    # This makes sense since dense retriever struggle with negation, but are otherwise fine.
    # The current appraoch does not use the power of dense retrievers.

    # Current chunking for ing does not carry semantic relevance.

    # set experiment
    mlflow.set_experiment("first-query-decomp-experiment")
    
    with mlflow.start_run():    
        # query and params
        query = "chicken"
        params = {
            "query": query,
            "retriever_top_k": 50,
            "reranker": "BAAI/bge-reranker-v2-m3",
            "reranker_max_length": 512,
            "reranker_top_k": 10,
        }

        # load recipes
        with open("data/processed/cleaned_recipes.json") as f:
            all_recipes = json.load(f)

        # load snapshot
        snapshot = load_snapshot("snapshots/latest")

        decomposed_query = decompose_query(query)
        inclusions = decomposed_query.split("Inclusions:")[1].split("Exclusions:")[0].strip()
        exclusions = decomposed_query.split("Exclusions:")[1].strip()  

        mlflow.log_dict(
            {
                "original_query": query,
                "inclusions": inclusions,
                "exclusions": exclusions
            },
            "decomposition.json"
        )

        print(f"Inclusions: {inclusions} \nExclusions: {exclusions}")
        
        # load retriever
        retriever = load_retriever(snapshot)

        # retrieve inclusions and exclusions
        inc_results = retriever.retrieve(inclusions, params["retriever_top_k"])
        exc_results = retriever.retrieve(exclusions, params["retriever_top_k"])
        
        # free memory
        del retriever
        torch.cuda.empty_cache()

        # extract recipe ids
        inc_recipe_ids = {r.chunk.recipe_id for r in inc_results}
        exc_recipe_ids = {r.chunk.recipe_id for r in exc_results}

        # deduplicate, remove excluded recipes
        recipe_ids = list(inc_recipe_ids - exc_recipe_ids)

        # pull recipes via ids
        candidate_recipes = [all_recipes[recipe_id] for recipe_id in recipe_ids]
        
        # format recipes for reranker
        reranker_inputs = [recipe_to_string(r) for r in candidate_recipes]

        # load reranker
        reranker = CrossEncoder(
            model_name_or_path=params["reranker"],
            max_length=params["reranker_max_length"]
        )

        # rerank
        ranked_results = reranker.rank(
            query=query,
            documents=reranker_inputs,
            return_documents=True,
            top_k=params["reranker_top_k"]
        )
        
        retrieved_data = []

        for r in inc_results:
            retrieved_data.append({
                "type": "inclusion",
                "recipe_id": r.chunk.recipe_id,
                "score": getattr(r, "score", None),
                "recipe_name": all_recipes[r.chunk.recipe_id]["recipe_name"],
                "text": recipe_to_string(all_recipes[r.chunk.recipe_id])
            })

        for r in exc_results:
            retrieved_data.append({
                "type": "exclusion",
                "recipe_id": r.chunk.recipe_id,
                "score": getattr(r, "score", None),
                "recipe_name": all_recipes[r.chunk.recipe_id]["recipe_name"],
                "text": recipe_to_string(all_recipes[r.chunk.recipe_id])
            })

        retrieved_df = pd.DataFrame(retrieved_data)

        filtered_data = [
            {
                "recipe_id": recipe_id,
                "recipe_name": all_recipes[recipe_id]["recipe_name"],
                "text": recipe_to_string(all_recipes[recipe_id])
            }
            for recipe_id in recipe_ids
        ]

        reranked_results = {
            "query": [query] * len(ranked_results),
            "rank": [i + 1 for i in range(len(ranked_results))],
            "reranker_score": [float(r["score"]) for r in ranked_results],
            "text": [r["text"] for r in ranked_results],
            "recipe_id": [
                candidate_recipes[r["corpus_id"]]["recipe_id"]
                for r in ranked_results
            ],
            "recipe_name": [
                candidate_recipes[r["corpus_id"]]["recipe_name"]
                for r in ranked_results
            ]    
        }

        reranked_df = pd.DataFrame(reranked_results)

        mlflow.log_params(params)
        mlflow.log_table(data=retrieved_df, artifact_file="retrieved_results.json")
        mlflow.log_table(data=pd.DataFrame(filtered_data), artifact_file="filtered_candidates.json")
        mlflow.log_table(data=reranked_df, artifact_file="results.json")

    
if __name__ == "__main__": 
    test_query_decomp()
    #test_recipe_to_string()
    #test_reranker()
