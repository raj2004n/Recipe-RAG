from src.retrieval.retriever import Retriever
from src.retrieval.encoder import Encoder
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.retrieval.indexer import Indexer
from transformers import AutoTokenizer
from src.data_classes import Snapshot
from typing import Dict

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
            max_context_length=snapshot.prompts_config["settings"]["max_context_tokens"],
            prompt_templates=snapshot.prompts_config["template"]
        )
        self.generator = Generator(
            snapshot.base_config["generator"]["model"],
            snapshot.base_config["generator"]["adapter_path"]
        )

    def query(self, query: str, k: int = 3) -> Dict:
        """Full RAG query: retrieve + generate"""
        results = self.retriever.retrieve(query, k)
        
        # format prompt with retrieved context
        prompt = self.prompt_builder.build_prompt(query, retrieval_results=results)
        
        # generate answer
        answer = self.generator.generate(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "prompt_used": prompt
        }