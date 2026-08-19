""" 
Dense Pipeline: dense retrieval, prompt, generate.


Note to self: rename to baseline or naive or something in the future, since
this is a very basic pipeline. 
"""


from transformers import AutoTokenizer
from ..retrieval.dense.retriever import RetrievalResult


GENERATION_PERSONA = "generation"


class PromptBuilder:
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            max_context_tokens: int,
            prompt_templates: dict
    ):
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

        Used by the DensePipeline. The chunks in retrieval results contain 
        the whole recipe formatted, so no recipes passed in.
        """
        context_parts = []
        current_tokens = 0

        for i, result in enumerate(retrieval_results):
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


    def build_prompt_from_recipes(
        self,
        query: str,
        recipes: list[dict],
        recipe_to_string,
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair from whole recipe dicts.

        Used by HybridPipeline after rerank: chunks are resolved back to
        whole recipes before the prompt is built, so the LLM gets full
        recipes instead of fragments.

        `recipe_to_string` is injected rather than imported so this module
        stays decoupled from the data layer and the same helper can be
        reused by notebooks.
        """
        context_parts = []
        current_tokens = 0

        for i, recipe in enumerate(recipes):
            text = recipe_to_string(recipe)

            tokens = self.tokenizer.encode(text)
            if current_tokens + len(tokens) > self.max_context_tokens:
                print(f"Max context tokens exceeded, breaking at recipe {i}")
                break

            context_parts.append(text)
            current_tokens += len(tokens)

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
