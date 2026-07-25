from rag.retrieval.dense.retriever import Retriever
from rag.retrieval.dense.encoder import Encoder
from src.rag.generation.prompt_builder import PromptBuilder
from src.rag.generation.generator import Generator
from rag.retrieval.dense.indexer import Indexer
from transformers import AutoTokenizer
from src.rag.core.snapshot import Snapshot
from typing import Dict
from src.rag.core.snapshot import RetrievalResult

class RAGPipeline:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        encoder = Encoder(
            snapshot.base_config["encoder"]["model"],
            snapshot.base_config["encoder"]["batch_size"],
            snapshot.base_config["encoder"]["normalize_embeddings"]
        )
        indexer = Indexer.load(
            snapshot.base_config["paths"]["files"]["indexer"],
            snapshot.base_config["indexer"]["type"],
            snapshot.base_config["indexer"]["dimension"]
        )
        self.retriever = Retriever(
            snapshot.chunks,
            encoder,
            indexer,
            snapshot.base_config["retriever"]["top_k"],
            snapshot.base_config["retriever"]["score_threshold"]
        )
        self.prompt_builder = PromptBuilder(
            tokenizer=AutoTokenizer.from_pretrained(snapshot.base_config["generator"]["model"]),
            max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
            prompt_templates=snapshot.prompts_config["template"]
        )
        self.generator = Generator(
            snapshot.base_config["generator"]["model"],
            snapshot.base_config["generator"]["max_input_tokens"],
            snapshot.base_config["generator"]["max_new_tokens"],
            snapshot.base_config["generator"]["truncation"],
            snapshot.base_config["generator"]["return_tensors"],
            snapshot.base_config["generator"]["adapter_path"]
        )

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