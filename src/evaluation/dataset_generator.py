import json
from pathlib import Path
from ollama import chat

MODEL_NAME = "gemma3:4b"

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

Format:
- Return a JSON array of strings. 

Recipe name:
{recipe["recipe_name"]}

Ingredients:
{ingredients}
"""
    return prompt

def generate_queries(recipe: dict) -> list[str]:
    """ Generates a list of queries for a recipe."""
    prompt = build_prompt(recipe)
    
    response = chat(
        MODEL_NAME,
        messages=[{"role": "user", "content": prompt,}]
    )

    content = response.message.content.strip()

    if content.startswith("```json"):
        content = content[7:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()

    try:
        queries = json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse JSON.")
        print(content)
        return []

    return queries

def generate_dataset(recipes_file: Path, output_file: Path):
    """ Generates a dataset of queries for a list of recipes."""
    with open(recipes_file, "r") as f:
        recipes = json.load(f)
    
    evaluation_dataset = []
    for recipe_id, recipe in enumerate(recipes):
        print(f"Generating queries for recipe {recipe_id}: {recipe['recipe_name']}")
        queries = generate_queries(recipe)

        for query in queries:
            evaluation_dataset.append(
                {
                    "recipe_id": recipe_id,
                    "recipe_name": recipe["recipe_name"],
                    "query": query,
                }
            )

    with open(output_file, "w") as f:
        json.dump(evaluation_dataset, f, indent=4)
    
    print(f"Saved {len(evaluation_dataset)} evaluation queries")
        