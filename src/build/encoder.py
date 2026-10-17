from sentence_transformers import SentenceTransformer
from ..domain.types import ChunkSet, EncodedChunk, EncodedChunkSet
import numpy as np


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

    def encode_chunks(self, chunks_set_list: list[ChunkSet]) -> list[EncodedChunkSet]:
        """ Encode list of chunks into dense vectors. """
        encoded_chunks_sets : list[EncodedChunkSet] = []
        for chunks_set in chunks_set_list:
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

            # name is kept the same as chunk set name
            # "encoded_chunks" is added at the end when saving
            encoded_chunks_sets.append(
                EncodedChunkSet(
                    name=chunks_set.name,
                    encoded_chunks=encoded_chunks
                )
            )
        return encoded_chunks_sets

    def encode_query(self, query: str) -> np.ndarray:
        """ Encode a single query into a dense vector. """
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding
    
    # Note: no need to save / load a pretrained model