from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from .chunker import Chunk
import numpy as np

@dataclass
class EncodedChunk:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, any]

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: list[float]) -> "EncodedChunk":
        return cls(
            id=f"recipe_{chunk.recipe_id}_chunk_{chunk.chunk_id}",
            vector=vector,
            text=chunk.text,
            metadata={
                "recipe_id": chunk.recipe_id,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type
            }
        )

class Encoder:
    def __init__(self, model, batch_size, normalize_embeddings):
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        """ Encode a list of documents into dense vectors. """
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings
    
    def encode_chunks(self, chunks: list[Chunk]) -> list[EncodedChunk]:
        """ Encode list of chunks into dense vectors. """
        texts = [c.text for c in chunks]
        embeddings = self._encode_documents(texts)

        encoded_chunks = [
            EncodedChunk.from_chunk(
                chunk=chunk,
                vector=embedding.tolist()
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        return encoded_chunks

    def encode_query(self, query: str) -> np.ndarray:
        """ Encode a single query into a dense vector. """
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding
    
    # Note: no need to save / load a pretrained model