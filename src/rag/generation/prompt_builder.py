from ..retrieval.dense.retriever import RetrievalResult
from transformers import AutoTokenizer

# Persona name that build_prompt resolves against. Defined centrally so
# the new prompts_config shape (template.personas.<name>.<generate|judge>.<user>)
# is documented in one place.
GENERATION_PERSONA = "generation"


class PromptBuilder:
    def __init__(self, tokenizer: AutoTokenizer, max_context_tokens: int, prompt_templates: dict):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.prompt_templates = prompt_templates

    def _persona_template(self, persona: str, role: str, body: str) -> str:
        """Look up a system or user body of a persona's generate or judge template.

        The prompts_config shape is:
            template:
              personas:
                <persona_name>:
                  generate: { system, user }
                  judge:    { system, user }

        body is "system" or "user".
        """
        try:
            return self.prompt_templates["personas"][persona][role][body]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"prompts_config is missing personas.{persona}.{role}.{body}; "
                f"check configs/prompts_config.yaml"
            ) from exc


    def build_prompt(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair for the generation persona.

        Returns a tuple so the caller (Generator) can construct a proper
        chat-template message list with both roles
        """
        context_parts = []
        current_tokens = 0

        for i, result in enumerate(retrieval_results):
            # Chunk is a dataclass (see chunker.py), not a dict.
            chunk_text = result.chunk.text

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

        user_prompt = self._persona_template(
            GENERATION_PERSONA, "generate", "user"
        ).format(context=context, query=query)
        system_prompt = self._persona_template(
            GENERATION_PERSONA, "generate", "system"
        )

        return system_prompt, user_prompt


    def build_judge_prompt(self, recipe: dict, query: str, response: str) -> str:
        """Build a prompt-ready context string for LLM judge during evaluation"""
        igredients = recipe.get("ingredients", [])
        ingredients = "\n".join(igredients)

        prompt_template = self._persona_template(
            GENERATION_PERSONA, "judge", "user"
        )

        return prompt_template.format(
            query=query,
            recipe=recipe,
            ingredients=ingredients,
            response=response,
        )
