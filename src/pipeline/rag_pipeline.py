# src/pipeline/rag_pipeline.py
from src.models.generator import Generator
from src.models.retriever import Retriever
from typing import Dict

class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
    
    def query(self, question: str, k: int = 3) -> Dict:
        """Full RAG query: retrieve + generate"""
        # format prompt with retrieved context
        prompt = self.retriever.format_for_llm(question, k)
        
        # generate answer
        answer = self.generator.generate(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "prompt_used": prompt
        }