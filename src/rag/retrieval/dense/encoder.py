from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Callable
from ..chunker import Chunk, ChunkSet
import numpy as np

@dataclass
class EncodedChunk:
    vector: list[float]
    text: str
    recipe_id: int
    chunk_id: int

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: list[float]) -> "EncodedChunk":
        return cls(
            vector=vector,
            text=chunk.text,
            recipe_id=chunk.recipe_id,
            chunk_id=chunk.chunk_id
        )

class Encoder:
    def __init__(
        self,
        model_id: str,
        batch_size: int,
        normalize_embeddings: bool
    ):
        self.model = SentenceTransformer(model_id)
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

    def encode_chunks(self, chunks_set_list: list[Chunk]) -> list[EncodedChunk]:
        """ Encode list of chunks into dense vectors. """
        # filter chunk sets 
        selected = [s for s in chunks_set_list if self.accepts(s)]

        for chunks_set in selected:
            chunks = chunks_set.chunks
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