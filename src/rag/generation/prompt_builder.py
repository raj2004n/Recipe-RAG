"""Build the (system, user) prompt pair the generator answers from.

Whole recipes and whole-recipe chunk text are two different starting points, so the pair is built two
ways. The hybrid path renders recipes into the student's own prompt: the system line the snapshot's
prompts config carries and a user turn carrying the recipes and the request, laid out by the renderer
the training rows were built from. The dense path has no recipe objects to render, because a
whole-recipe chunk carries the formatted text itself, so it fills the persona template from
configs/prompts_config.yaml with the text it retrieved.

Either way the context is held to a token budget: a recipe that would take the turn past the ceiling
ends the list, so what a prompt carries is always a prefix of the ranked recipes.
"""


from transformers import AutoTokenizer
from ..retrieval.dense.retriever import RetrievalResult
from ..data.preprocessor import Recipe
from ..data.source_block import labelled_blocks
from ..data.student_prompt import student_prompt, student_system


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


    def _kept_blocks(self, recipes: list[dict]) -> list[str]:
        """The rendered recipe blocks that fit the context budget, in rank order.

        The recipes arrive as corpus rows, the shape a pipeline indexes and an evaluator holds, so
        each is read as a Recipe before it is rendered. A block that would take the context past its
        ceiling ends the list rather than being trimmed, so the labels a turn carries always run from
        the first ranked recipe onward.
        """
        kept = []
        current_tokens = 0

        blocks = labelled_blocks([Recipe(**row) for row in recipes])
        for index, block in enumerate(blocks, start=1):
            tokens = self.tokenizer.encode(block)
            if current_tokens + len(tokens) > self.max_context_tokens:
                print(f"Max context tokens exceeded, breaking at recipe {index}")
                break

            kept.append(block)
            current_tokens += len(tokens)

        return kept


    def build_prompt_from_recipes(
        self,
        query: str,
        recipes: list[dict],
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair from whole recipes, as the student was trained.

        Used by HybridPipeline after rerank: chunks are resolved back to
        whole recipes before the prompt is built, so the model gets full
        recipes instead of fragments. The pair is the student's own rather than a persona's, since
        it is the prompt the adapter in the snapshot was fitted on, and the request is the whole of
        the user turn rather than an instruction wrapped around the recipes. Its system line is read
        from the snapshot's prompts config, which is the copy the build sealed.
        """
        return student_prompt(
            student_system(self.prompt_templates),
            self._kept_blocks(recipes),
            query,
        )
