import json
import argparse
from pathlib import Path
from ollama import chat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

MODEL_NAME = "gemma3:4b"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
BACKOFF_BASE = 2
MAX_WORKERS = 4

class QueryList(BaseModel):
    queries: list[str]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes_path", type=str, required=False, default="data/processed/cleaned_recipes.json")
    parser.add_argument("--output_path", type=str, required=False, default="data/processed/evaluation_queries.json")
    parser.add_argument("--max_workers", type=int, required=False, default=MAX_WORKERS)
    return parser.parse_args()


def get_ingredient_list(recipe: dict) -> list[str]:
    """ Returns a list of ingredients for a recipe."""
    ingredients = []

    for group in recipe["ingredient_groups"]:
        for ingredient in group["ingredients"]:
            ingredients.append(ingredient["name"])
    
    return ingredients

def build_prompt(recipe: dict) -> str:
    """ Builds a prompt for the LLM to generate a query for a recipe."""

    ingredients = ", ".join(get_ingredient_list(recipe))

    # currently only supports ingredients and not instructions

    # may consider ignoring adjectives from recipe names
    prompt = f"""
You are creating evaluation queries for a recipe retrieval system.

Generate three queries that a user may input to retrieve this recipe.

Reules:
- Each query should be unique.
- Each query should be under 20 words.
- Some queries should mention ingredients.
- Some queries should mention recipe name.
- Ignore adjectives in recipe names.
- Ignore people names in recipe names.

Respond only with JSON in this exact shape:
{{"queries": ["query 1", "query 2", "query 3"]}}

Recipe name:
{recipe["recipe_name"]}

Ingredients:
{ingredients}
"""
    return prompt

def generate_queries(recipe: dict) -> list[str]:
    """ Generates a list of queries for a recipe."""
    prompt = build_prompt(recipe)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = chat(
                MODEL_NAME,
                messages=[{"role": "user", "content": prompt,}],
                format="json"
            )
            parsed = QueryList.model_validate_json(response.message.content)
            return parsed.queries
        except ValidationError as e:
            last_error = e
        except Exception as e:
            last_error = e
        
        if attempt < MAX_RETRIES - 1:
            sleep_for = BACKOFF_BASE ** attempt
            time.sleep(sleep_for)

    return []

def process_recipe(recipe_id: int, recipe: dict) -> tuple[int, dict, list[str]]:
    queries = generate_queries(recipe)
    return recipe_id, recipe, queries

def load_processed_ids(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()
    processed = set()
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add(row["recipe_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return processed

def generate_dataset(recipes_file: Path, output_file: Path, max_workers: int = MAX_WORKERS):
    """ Generates a dataset of queries for a list of recipes."""
    with open(recipes_file, "r") as f:
        recipes = json.load(f)
    
    already_processed = load_processed_ids(output_file)
    todo = [(i, r) for i, r in enumerate(recipes) if i not in already_processed]

    if not todo:
        return
    
    failed_ids = []

    with open(output_file, "a") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_recipe, i, r): i for i, r in todo}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
            recipe_id = futures[fut]
            try:
                recipe_id, recipe, queries = fut.result()
            except Exception:
                failed_ids.append(recipe_id)
                continue

            if not queries:
                failed_ids.append(recipe_id)
                continue

            for q in queries:
                out_f.write(json.dumps({
                    "recipe_id": recipe_id,
                    "recipe_name": recipe["recipe_name"],
                    "query": q,
                }) + "\n")
            out_f.flush()
        
        if failed_ids:
            preview = failed_ids[:10]
            print(f"Failed to generate queries for {len(failed_ids)} recipes: {preview}")
    
    print(f"Saved {len(todo) - len(failed_ids)} evaluation queries")

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(Path(args.recipes_path), Path(args.output_path), max_workers=args.max_workers)