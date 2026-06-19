import yaml
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class RetrievalResult:
    chunk: dict
    score: float

class Retriever:
    def __init__(self, encoder, index, chunks: list[dict], config: dict):
        
        self.encoder = encoder
        self.index = index
        self.top_k = config["retrieval"]["top_k"]
        self.chunks = chunks
        self.score_threshold = config["retrieval"]["score_threshold"]

        self._chunk_id_to_idx: dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("id") or chunk.get("id") or str(i)
            self._chunk_id_to_idx[chunk_id] = i
    
    def retrieve(self, query: str, top_k : int = None) -> list[RetrievalResult]:
        
        top_k = top_k or self.top_k
        
        # encode query
        query_vector = self.encoder.encode_query(query)

        # search index
        scores, indices = self.index.search(query_vector, top_k)

        # build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # skip invlaid indices
            if idx == -1 or idx >= len(self.chunks):
                continue
            # filter by score
            if self.score_threshold and score < self.score_threshold:
                continue
        
            chunk = self.chunks[idx]
            
            results.append(RetrievalResult(
                chunk=chunk,
                score=float(score),
            ))

            if len(results) >= top_k:
                break
        
        return results

    def retrieve_as_text(self, query: str, top_k: int = None) -> list[str]:
        """ Retrieve and return chunk texts directly. """
        results = self.retrieve(query, top_k)
        return [r.chunk["text"] for r in results]
    
    def format_for_llm(self, query: str, top_k: int = None, system_prompt: str = None, max_context_lenght: int = 1500) -> str:
        """ Build a prompt-ready context string. """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            meta = result.chunk.get("metadata", {})
            recipe_name = meta.get("recipe_name", f"Document {i}")
            chunk_text = f"[Recipe {i}: {recipe_name}]\n{result.chunk['text']}"
  
            if current_length + len(chunk_text) > max_context_lenght:
                break
        
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful cooking assistant. "
                "Answer the user's question using ONLY the recipes provided below. "
                "If the recipes don't contain enough information, say so honestly. "
                "Always mention which recipe(s) you're referencing."
            )
        return f"""{system_prompt}
    
RECIPES:
{context}

USER QUESTION: {query}

ANSWER:"""
    
    def save(self, path: str):
        """ Save index and chunks to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        self.index.save(os.path.join(path, "index.faiss"))
        
        # Save chunks
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved retriever to {path}")
    
    @classmethod
    def load(cls, path: str, encoder, indexer, config_path: str):
        """ Load a saved retriever. """
        # Load chunks
        with open(os.path.join(path, "chunks.json"), "r") as f:
            chunks = json.load(f)
        
        # Load index
        indexer.load(os.path.join(path, "index.faiss"))
        
        return cls(
            encoder=encoder,
            indexer=indexer,
            chunks=chunks,
            config_path=config_path,
            index_path=path
        )