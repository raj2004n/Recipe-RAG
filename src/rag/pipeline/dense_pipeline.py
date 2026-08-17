from ..retrieval.dense.retriever import Retriever, RetrievalResult
from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..generation.prompt_builder import PromptBuilder
from ..generation.generator import Generator
from transformers import AutoTokenizer
from ..core.snapshot import Snapshot


class DensePipeline:
    def __init__(self, snapshot: Snapshot):
        assert self._correct_strategy, "Invalide snapshot. DensePipeline requires the dense strategy"
        
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

        self.prompt_builder = PromptBuilder(
            tokenizer=AutoTokenizer.from_pretrained(snapshot.base_config["generator"]["model_id"]),
            max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
            prompt_templates=snapshot.prompts_config["template"]
        )

        self.generator = Generator(
            snapshot.base_config["generator"]["model_id"],
            snapshot.base_config["generator"]["max_input_tokens"],
            snapshot.base_config["generator"]["max_new_tokens"],
            snapshot.base_config["generator"]["truncation"],
            snapshot.base_config["generator"]["return_tensors"],
            snapshot.base_config["generator"]["adapter_path"]
        )


    def _correct_strategy(self, snapshot: Snapshot) -> bool:
        if snapshot.strategy != "dense":
            return False
        return True


    def query(self, query: str, k: int = 3):
        """Full RAG query: retrieve + generate"""
        results = self.retriever.retrieve(query, k)
        
        # format prompt with retrieved context
        prompt = self.prompt_builder.build_prompt(query, retrieval_results=results)
        
        # generate answer
        answer = self.generator.generate(prompt)
        
        # query, and prompt used are not used in this method
        return {
            "query": query,
            "answer": answer,
            "prompt_used": prompt
        }


    def debug_query(self, query: str, k: int = 3) -> tuple[list[RetrievalResult], str, str]:
        """ Query used for debugging purposes, returns retrieved results, query, and prompt used"""
        results = self.retriever.retrieve(query, k)
        
        # format prompt with retrieved context
        prompt = self.prompt_builder.build_prompt(query, retrieval_results=results)
        
        # generate answer
        #response = self.generator.generate(prompt)
        response = ""
        return results, response, prompt