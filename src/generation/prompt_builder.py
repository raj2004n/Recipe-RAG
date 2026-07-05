from src.retrieval.retriever import RetrievalResult
from transformers import AutoTokenizer

#TODO: Change structure
# This is in its own dir, but not needed

class PromptBuilder:
    def __init__(self, tokenizer: AutoTokenizer, max_context_length: int):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
    
    def build_prompt(self, prompt_template: str, query: str, retrieval_results: list[RetrievalResult]) -> str:
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
        
        return prompt_template.format(
            context=context,
            query=query
        )
    
    def build_judge_prompt(self, prompt_template: str, recipe: dict, query: str, response: str) -> str:
        """ Build a prompt-ready context string for LLM judge during evaluation. """
        igredients = recipe.get("ingredients", [])
        ingredients = "\n".join(igredients)

        return prompt_template.format(
            query=query,
            recipe=recipe,
            ingredients=ingredients,
            response=response
        )
