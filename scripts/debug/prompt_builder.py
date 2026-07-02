from src.prompting.prompt_builder import PromptBuilder
from src.models.retriever import RetrievalResult

# hard code to test
config = {
    "prompts": {
        "system": "You are a helpful cooking assistant. Answer the user's question using ONLY the recipes provided below. If the recipes don't contain enough information, say so honestly. Always mention which recipe(s) you're referencing.",
        "max_context_length": 1500,
    }
}

builder = PromptBuilder(config)

results = [
    RetrievalResult(
        chunk_id="1_0",
        recipe_id="1",
        chunk={
            "text": "Recipe: Chicken Curry...",
            "metadata": {
                "recipe_name": "Chicken Curry"
            }
        },
        score=0.9
    )
]

prompt = builder.build(
    query="How do I make chicken curry?",
    retrieval_results=results
)

print(prompt)