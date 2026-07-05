from sentence_transformers import SentenceTransformer
import numpy as np

class Encoder:
    def __init__(self, model, batch_size, normalize_embeddings):
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """ Encode a list of documents into dense vectors. """
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """ Encode a single query into a dense vector. """
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding

    # no need to save / load a pretrained model