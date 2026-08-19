"""
Pipeline classes are used to load the model snapshot and run inference.

The chunks passed in here contain formatted whole recipes.
Thus, no formatting step is required as the one present in Hybrid Pipeline.
"""

from ..retrieval.dense.retriever import Retriever, RetrievalResult
from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..generation.prompt_builder import PromptBuilder
from ..generation.generator import Generator
from transformers import AutoTokenizer
from ..core.snapshot import Snapshot


class DensePipeline:
    ALLOWED_STRATEGIES = ("dense",)

    def __init__(self, snapshot: Snapshot):
        if snapshot.retrieval_strategy not in self.ALLOWED_STRATEGIES:
            raise ValueError(
                f"DensePipeline requires retrieval_strategy in "
                f"{self.ALLOWED_STRATEGIES}, got "
                f"{snapshot.retrieval_strategy!r}"
            )

        # lets load_generator read generator config later
        self._snapshot = snapshot

        encoder = Encoder(
            model_id=snapshot.base_config["encoder"]["model_id"],
            batch_size=snapshot.base_config["encoder"]["batch_size"],
            normalize_embeddings=snapshot.base_config["encoder"]["normalize_embeddings"],
        )

        indexer = Indexer(
            index_type=snapshot.base_config["indexer"]["type"],
            dimension=snapshot.base_config["indexer"]["dimension"],
            score_threshold=snapshot.base_config["retriever"]["score_threshold"],
        )
        # IndexedChunkSet.chunks was populated by load_snapshot, so the
        # indexer has everything it needs to resolve hits.
        indexer.set_indexed_sets(snapshot.indexed_chunk_sets)

        self.retriever = Retriever(
            encoder=encoder,
            indexer=indexer,
            top_k=snapshot.base_config["retriever"]["top_k"],
        )

        # load the tokenizer once and share it with both PromptBuilder
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot.base_config["generator"]["model_id"]
        )

        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
            prompt_templates=snapshot.prompts_config["template"]
        )

        self.generator = None


    def load_generator(self) -> Generator:
        """Lazily construct and cache the Generator from snapshot config"""
        if self.generator is None:
            gen_cfg = self._snapshot.base_config["generator"]
            self.generator = Generator(
                model_id=gen_cfg["model_id"],
                max_input_tokens=gen_cfg["max_input_tokens"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                truncation=gen_cfg["truncation"],
                return_tensors=gen_cfg["return_tensors"],
                adapter_path=gen_cfg["adapter_path"],
                tokenizer=self.tokenizer,
            )
        return self.generator


    def query(self, query: str, k: int = 3):
        """Full RAG query: retrieve + generate"""
        results = self.retriever.retrieve(query, k)

        # build_prompt returns (system, user) for chat-template generation
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query, retrieval_results=results
        )

        # generate answer (loads the model on first call, cached afterwards)
        answer = self.load_generator().generate(system_prompt, user_prompt)

        return {
            "query": query,
            "answer": answer,
            "prompt_used": (system_prompt, user_prompt),
        }


    def debug_query(self, query: str, k: int = 3) -> tuple[list[RetrievalResult], str, tuple[str, str]]:
        results = self.retriever.retrieve(query, k)

        prompts = self.prompt_builder.build_prompt(
            query, retrieval_results=results
        )

        return results, "", prompts