from src.prompting.prompt_builder import PromptBuilder
from src.models.retriever import RetrievalResult
from transformers import AutoTokenizer

# hard code to test
config = {
    "generator": {
        "model_name": "microsoft/phi-2"
    },
    "prompts": {
        "system": "You are a helpful cooking assistant. Answer the user's question using ONLY the recipes provided below. If the recipes don't contain enough information, say so honestly. Always mention which recipe(s) you're referencing.",
        "max_context_tokens": 1500,
    }
}

tokenizer = AutoTokenizer.from_pretrained(config["generator"]["model_name"])

builder = PromptBuilder(config, tokenizer)

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