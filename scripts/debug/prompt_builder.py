from src.prompting.prompt_builder import PromptBuilder
from src.models.retriever import RetrievalResult
from transformers import AutoTokenizer
from src.utils import load_config

config = load_config()

tokenizer = AutoTokenizer.from_pretrained(config["generator"]["model_name"])

builder = PromptBuilder(config, tokenizer)

results = [
    RetrievalResult(
        chunk_id="1_0",
        recipe_id="1",
        chunk={
            "text": "Chicken Curry recipe...",
        },
        score=0.9
    ),
    RetrievalResult(
        chunk_id="2_0",
        recipe_id="2",
        chunk={
            "text": "Chicken and Mushroom Curry recipe...",
        },
        score=0.8
    ),
]

prompt = builder.build(
    query="How do I make chicken curry?",
    retrieval_results=results
)

print(prompt)