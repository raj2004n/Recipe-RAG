import faiss
import numpy as np
from ..schemas import EncodedChunk

class Indexer:
    def __init__(self, index_type: str, dimension: int):
        self.index_type = index_type
        self.dimension = dimension
        self.index = self._create_index()
        self.id_mapping: dict[int,str] = {}
    
    def _create_index(self) -> faiss.Index:
        """ Create a new index of the specified type. """
        if self.index_type == "FlatIP":
            return faiss.IndexFlatIP(self.dimension)
        
        raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _add(self, vectors: np.ndarray, ids: np.ndarray):
        """ Add vectors to the index. """
        # numpy defualt is float64, but faiss requires float32
        vectors = vectors.astype("float32")
        ids = np.asarray(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids)

    def index_embeddings(self, chunks: list[EncodedChunk]):
        """ Assign ids to the embeddings. """
        vectors = np.asarray(
            [c.vector for c in chunks],
            dtype="float32"
        )
        # FAISS needs int ids
        faiss_ids = np.arange(len(chunks), dtype=np.int64)

        # keep mapping back to original chunk ids
        for faiss_id, chunk in zip(faiss_ids, chunks):
            self.id_mapping[int(faiss_id)] = chunk.id

        self._add(vectors, faiss_ids)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, list[list[str]]]:
        """ Search and return original chunk ids. """
        query_vectors = query_vectors.astype("float32")

        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # in order of similarity
        scores, indices = self.index.search(query_vectors, k)
        
        chunk_ids = [
            [
                self.id_mapping.get(int(idx)) for idx in row if idx != -1
            ]
            for row in indices
        ]

        return scores, indices

    def save(self, path: str):
        """ Save the index to disk. """
        faiss.write_index(self.index, path)

        np.save(f"{path}.mapping.npy", self.id_mapping, allow_pickle=True)

    @classmethod
    def load(cls, path: str, index_type: str, dimension: int) -> "Indexer":
        """ Load the index from disk. """
        indexer = cls(index_type, dimension)
        indexer.index = faiss.read_index(path)
        indexer.id_mapping = np.load(
            f"{path}.mapping.npy",
            allow_pickle=True,
        ).item()
        
        return indexer