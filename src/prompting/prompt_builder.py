from src.models.retriever import RetrievalResult
from transformers import AutoTokenizer

class PromptBuilder:
    def __init__(self, config: dict, tokenizer: AutoTokenizer):
        self.system_prompt = config["prompts"]["system"]
        self.max_context_length = config["prompts"]["max_context_tokens"]
        self.tokenizer = tokenizer
    
    def build(self, query: str, retrieval_results: list[RetrievalResult]) -> str:
        """ Build a prompt-ready context string. """
        context_parts = []
        current_length = 0

        for i, result in enumerate(retrieval_results):
            chunk_text= result.chunk["text"]

            tokens = self.tokenizer.encode(chunk_text)

            if current_length + len(tokens) > self.max_context_length:
                print(f"Context length exceeded, breaking at chunk {i}")
                break
        
            context_parts.append(chunk_text)

            current_length += len(tokens)

        context = "\n\n".join(context_parts)
        
        return f"""{self.system_prompt}
    
Recipes: \n{context}

User Query: \n{query}"""
    
    def build_judge_prompt(self, recipe: dict, query: str, response: str) -> str:
        """ Build a prompt-ready context string for LLM judge during evaluation. """
        igredients = recipe.get("ingredients", [])
        ingredients = "\n".join(igredients)

        return f"""
You are a judge for a recipe RAG model.

User Query: {query}

Retrieved Recipe: 
Name: {recipe["name"]}
Ingredients: {ingredients}
Instructions: {recipe["instructions"]}

Response: {response}

Score the following:
1. Query relevance (0 - 2)
0 = irrelevant
1 = partially relevant
2 = relevant

2. Faithfulness (0 - 2)
0 = irrelevant
1 = partially relevant
2 = relevant

Return JSON:
{{
    "query_relevance": int,
    "faithfulness": int
}}
"""
