from dataclasses import dataclass
from .encoder import Encoder
from .indexer import Indexer
import json
import os

@dataclass
class RetrievalResult:
    chunk_id: str
    recipe_id: str
    chunk: dict
    score: float

class Retriever:
    def __init__(self, encoder: Encoder, index: Indexer, chunks: list[dict], config: dict):
        """ Initialise a Retriever object.

            Args:
                encoder (Encoder): The encoder object used to encode documents.
                index (Indexer): The index object used to search for nearest neighbors.
                chunks (list[dict]): A list of dictionaries containing chunk information.
                config (dict): The configuration dictionary.

        The encoder turns the query into a dense vector in the same space as the recipe vectors.
        The index is used to search for the nearest neighbors of the query vector (in the encoder space of the recipe vectors).
        The chunks maps the retrieved indices (top k) to the corresponding recipe chunks.
        """
        self.encoder = encoder
        self.index = index
        self.top_k = config["retrieval"]["top_k"]
        self.chunks = chunks
        self.score_threshold = config["retrieval"]["score_threshold"]
    
    def retrieve(self, query: str, top_k : int = None) -> list[RetrievalResult]:
        top_k = top_k or self.top_k
        
        # encode query into dense vector
        query_vector = self.encoder.encode_query(query)

        # search recipe index for nearest neighbors of query vector
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
            
            # assign chunk to nearest neighbor recipe chunk
            chunk = self.chunks[idx]
            
            # add result to list
            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                recipe_id=chunk["metadata"]["recipe_id"],
                chunk=chunk,
                score=float(score),
            ))

            # break if enough results are found
            if len(results) >= top_k:
                break
        
        return results

    def retrieve_as_text(self, query: str, top_k: int = None) -> list[str]:
        """ Retrieve and return chunk texts directly. """
        results = self.retrieve(query, top_k)
        return [r.chunk["text"] for r in results]
    
    # TODO: seperate into a different file
    def format_for_llm(self, query: str, top_k: int = None, system_prompt: str = None, max_context_length: int = 1500) -> str:
        """ Build a prompt-ready context string. """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            meta = result.chunk.get("metadata", {})
            recipe_name = meta.get("recipe_name", f"Document {i}")
            chunk_text = f"[Recipe {i}: {recipe_name}]\n{result.chunk['text']}"
  
            if current_length + len(chunk_text) > max_context_length:
                break
        
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # TODO: save system prompt in config and track with ml flow
        if system_prompt is None:
            system_prompt = (
                "You are a helpful cooking assistant. "
                "Answer the user's question using ONLY the recipes provided below. "
                "If the recipes don't contain enough information, say so honestly. "
                "Always mention which recipe(s) you're referencing."
            )
        return f"""{system_prompt}
    
RECIPES: {context}

USER QUESTION: {query}

ANSWER:"""
    #TODO: Change "out" field of dvc.yaml once save modified
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
    def load(cls, path: str, encoder, index, config: dict):
        """ Load a saved retriever. """
        # Load chunks
        with open(os.path.join(path, "chunks.json"), "r") as f:
            chunks = json.load(f)
        
        # Load index
        index.load(os.path.join(path, "index.faiss"))
        
        return cls(
            encoder=encoder,
            index=index,
            chunks=chunks,
            config=config,
        )