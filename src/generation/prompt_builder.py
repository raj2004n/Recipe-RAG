from src.retrieval.retriever import RetrievalResult
from transformers import AutoTokenizer

class PromptBuilder:
    def __init__(self, tokenizer: AutoTokenizer, max_context_tokens: int, prompt_templates: dict):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.prompt_templates = prompt_templates
    
    def build_prompt(self, query: str, retrieval_results: list[RetrievalResult]) -> str:
        """ Build a prompt-ready context string. """
        context_parts = []
        current_tokens = 0

        for i, result in enumerate(retrieval_results):
            chunk_text= result.chunk["text"]

            # break if context length exceeded
            tokens = self.tokenizer.encode(chunk_text)
            if current_tokens + len(tokens) > self.max_context_tokens:
                print(f"Max context tokens exceeded, breaking at chunk {i}")
                break
            
            # add chunk to context and increment length
            context_parts.append(chunk_text)
            current_tokens += len(tokens)

        # join context parts
        context = "\n\n".join(context_parts)

        prompt_template = self.prompt_templates["generation"]
        
        return prompt_template.format(
            context=context,
            query=query
        )
    
    def build_judge_prompt(self, recipe: dict, query: str, response: str) -> str:
        """ Build a prompt-ready context string for LLM judge during evaluation. """
        igredients = recipe.get("ingredients", [])
        ingredients = "\n".join(igredients)

        prompt_template = self.prompt_templates["judge"]

        return prompt_template.format(
            query=query,
            recipe=recipe,
            ingredients=ingredients,
            response=response
        )
