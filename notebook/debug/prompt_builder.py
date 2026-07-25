from src.rag.generation.prompt_builder import PromptBuilder
from rag.retrieval.dense.retriever import RetrievalResult
from transformers import AutoTokenizer
from src.rag.core.utils import load_config

prompts_config = load_config("configs/prompts_config.yaml")
templates = prompts_config["template"]
config = prompts_config["config"]

# TODO: Considering the tokenizer need to be the same (?), and just preloaded, make sure it is consisten or saved
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

builder = PromptBuilder(
    tokenizer,
    config["max_context_tokens"],
    templates,
)

# made up results
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


prompt = builder.build_prompt(
    query="How do I make chicken curry?",
    retrieval_results=results
)

print(prompt)