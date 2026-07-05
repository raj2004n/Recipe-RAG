from src.models.retriever import Retriever
from src.prompting.prompt_builder import PromptBuilder
from src.models.generator import Generator
from transformers import AutoTokenizer
from typing import Dict

class RAGPipeline:
    def __init__(self, retriever: Retriever, prompt_builder: PromptBuilder, generator: Generator):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def query(self, query: str, k: int = 3) -> Dict:
        """Full RAG query: retrieve + generate"""
        results = self.retriever.retrieve(query, k)
        
        # format prompt with retrieved context
        prompt = self.prompt_builder.build(query, retrieval_results=results)
        
        # generate answer
        answer = self.generator.generate(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "prompt_used": prompt
        }