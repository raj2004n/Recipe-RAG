import json
import mlflow
import pandas as pd
from src.rag.core.snapshot import load_snapshot
from src.rag.core.utils import load_retriever
from sentence_transformers import CrossEncoder

def recipe_to_string(recipe: dict) -> str:
    """Flattens the nested recipe JSON into a clean, searchable text block."""
    text_parts = [f"Recipe Name: {recipe['recipe_name']}"]
    
    # parse ingredients
    text_parts.append("Ingredients:")
    for group in recipe.get("ingredient_groups", []):
        for ing in group.get("ingredients", []):
            amt = f"{ing['amount']} {ing['unit']}".strip()
            notes = f" ({ing['notes']})" if ing['notes'] else ""
            text_parts.append(f"- {amt} {ing['name']}{notes}")
            
    # parse instructions
    text_parts.append("Instructions:")
    for group in recipe.get("instruction_groups", []):
        for step in group.get("instructions", []):
            text_parts.append(f"- {step}")
    
    # parse notes
    recipe_notes = recipe.get("notes", [])
    if recipe_notes:
        text_parts.append("Notes:")
        for note in recipe_notes:
            text_parts.append(f"- {note}")

    # parse servings
    if "servings" in recipe:
        text_parts.append(f"Servings: {recipe['servings']}")

    return "\n".join(text_parts)

def test_recipe_to_string():
    query = "Pork recipes with no potatoes"

    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)

    # load snapshot, retriever
    snapshot = load_snapshot("snapshots/latest")
    retriever = load_retriever(snapshot)

    retrieved_results = retriever.retrieve(query=query, top_k=5)
    recipe_ids = [r.chunk.recipe_id for r in retrieved_results]

    recipes = [recipes[recipe_id] for recipe_id in recipe_ids]

    formatted_recipes = [recipe_to_string(r) for r in recipes]
    print(formatted_recipes[0])
    print("\n\n")
    print(formatted_recipes[1])

def test_reranker():
    # consider chunk type...

    # params
    params = {
        "query": "Pork recipes with no potatoes",
        "retriever_top_k": 50,
        "reranker": "BAAI/bge-reranker-v2-m3",
        "reranker_max_length": 512,
        "reranker_top_k": 10,
    }
    query = "Pork recipes with no potatoes"

    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)

    # load snapshot, retriever
    snapshot = load_snapshot("snapshots/latest")
    retriever = load_retriever(snapshot)

    # load reranker
    reranker = CrossEncoder(
        model_name_or_path=params["reranker"],
        max_length=params["reranker_max_length"]
    )
    
    with mlflow.start_run():
        retrieved_results = retriever.retrieve(query=query, top_k=params["retriever_top_k"])
        recipe_ids = [r.chunk.recipe_id for r in retrieved_results]

        recipes = [recipes[recipe_id] for recipe_id in recipe_ids]

        # format recipes for reranker inputs
        reranker_inputs = [recipe_to_string(r) for r in recipes]

        ranked_results = reranker.rank(
            query=query,
            documents=reranker_inputs,
            return_documents=True,
            top_k=params["reranker_top_k"]
        )

        structured_data = {
            "query": [query] * len(ranked_results),
            "rank": [i + 1 for i in range(len(ranked_results))],
            "reranker_score": [float(r["score"]) for r in ranked_results],
            "text": [r["text"] for r in ranked_results],
            "recipe_id": [recipes[i]["recipe_id"] for i in range(len(ranked_results))],
            "recipe_name": [recipes[i]["recipe_name"] for i in range(len(ranked_results))]
        }
        
        df = pd.DataFrame(structured_data)

        mlflow.log_table(data=df, artifact_file="results.json")
        mlflow.log_params(params)
