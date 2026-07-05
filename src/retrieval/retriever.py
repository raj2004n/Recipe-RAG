from dataclasses import dataclass
from .encoder import Encoder
from .indexer import Indexer
import json
import os

# TODO: Should have save and load method here for consistency.
# Consider a serparate save methods for the index and the chunks for clarity.

@dataclass
class RetrievalResult:
    chunk_id: str
    recipe_id: str
    chunk: dict
    score: float

class Retriever:
    def __init__(
            self,
            chunks: list[dict],
            encoder: Encoder,
            indexer: Indexer,
            top_k: int,
            score_threshold: float
        ):
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
        self.chunks = chunks
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(self, query: str, top_k : int = None) -> list[RetrievalResult]:
        top_k = top_k or self.top_k
        
        # encode query into dense vector
        query_vector = self.encoder.encode_query(query)

        # search recipe index for nearest neighbors of query vector
        scores, indices = self.indexer.search(query_vector, top_k)

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
    
    def save_chunks(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
    
    def save_indexer(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.indexer.save(os.path.join(path, "index.faiss"))

    def load_chunks(self, path: str):
        with open(os.path.join(path, "chunks.json")) as f:
            chunks = json.load(f)
        return chunks 
    
    def load_indexer(self, path: str):
        self.indexer = Indexer(
            1, # placeholder, overwrite with load
            self.indexer.dimension
        )
        self.indexer.load(os.path.join(path, "index.faiss"))
        return self.indexer
    