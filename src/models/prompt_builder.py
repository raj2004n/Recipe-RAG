from src.models.retriever import RetrievalResult

class PromptBuilder:
    def __init__(self, config: dict):
        self.system_prompt = config["prompts"]["system"]
        self.max_context_length = config["prompts"]["max_context_length"]
    
    def build(self, query: str, retrieval_results: list[RetrievalResult]) -> str:
        """ Build a prompt-ready context string. """
        context_parts = []
        current_length = 0

        for i, result in enumerate(retrieval_results, 1):
            recipe_name = result.chunk["metadata"].get(
                "recipe_name",
                f"Recipe {result.recipe_id}"
            )
            chunk_text= (
                f"[Recipe {i}: {recipe_name}]\n"
                f"{result.chunk.get('text', '')}"
            )
        
            if current_length + len(chunk_text) > self.max_context_length:
                break
        
            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n\n---\n\n".join(context_parts)
        
        return f"""{self.system_prompt}
    
RECIPES: {context}

USER QUESTION: {query}

ANSWER:"""