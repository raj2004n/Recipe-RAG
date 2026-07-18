import json
import mlflow
from src.snapshot import load_snapshot
from src.retrieval.indexer import Indexer 
from scripts.debug.prompt_decomp import decompose_query
from src.utils import flatten_config_dict, load_retriever

#mlflow.set_tracking_uri("http://localhost:5000")
#amlflow.set_experiment("first-retriever-experiment")

def inspect_index(indexer: Indexer, chunks: list[dict]):
    # number of chunks
    print(f"Number of chunks: {len(chunks)}")
    
    # number of vectors
    print(f"Number of vectors: {indexer.index.ntotal}")

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

def test_query_decomp(queries: list[str]):
    #TODO retrieve 50 - 100 results
    
    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)

    # load snapshot, retriever
    snapshot = load_snapshot("snapshots/latest")
    retriever = load_retriever(snapshot)

    for q in queries:
        # decompose query into inclusions and exclusions
        decomposed_query = decompose_query(q)
        inclusions = decomposed_query.split("Inclusions:")[1].split("Exclusions:")[0].strip()
        exclusions = decomposed_query.split("Exclusions:")[1].strip()  

        print(f"Inclusions: {inclusions} \nExclusions: {exclusions}")

        # retriever inclusions and exclusions
        inc_results = retriever.retrieve(inclusions) # TODO: Split inclusions into separate prompts?
        exc_results = retriever.retrieve(exclusions)
        
        # extract recipe ids
        inc_recipe_ids = [r.chunk.recipe_id for r in inc_results]
        exc_recipe_ids = [r.chunk.recipe_id for r in exc_results]

        # combine
        recipe_ids = inc_recipe_ids + exc_recipe_ids

        # deduplicate
        recipe_ids = set(recipe_ids)

        # pull recipes via ids
        inc_recipes = [recipes[recipe_id]["recipe_name"] for recipe_id in inc_recipe_ids]
        exc_recipes = [recipes[recipe_id]["recipe_name"] for recipe_id in exc_recipe_ids]

        print(f"Inclusion Recipes: {inc_recipes} \nExclusion Recipes: {exc_recipes}")



if __name__ == "__main__": 
    queries = [
        "Pork recipes with no potatoes",
    ]
    test_query_decomp(queries)