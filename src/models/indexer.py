import faiss
import numpy as np

class Indexer:
    def __init__(self, index_type: str, dimension: int):
        self.index_type = index_type
        self.dimension = dimension
        self.index = self._create_index()
    
    def _create_index(self) -> faiss.Index:
        """ Create a new index of the specified type. """
        if self.index_type == "FlatIP":
            return faiss.IndexFlatIP(self.dimension)
        
    def add(self, vectors: np.ndarray, ids: np.ndarray):
        """ Add vectors to the index. """
        # numpy defualt is float64, but faiss requires float32
        vectors = vectors.astype("float32")

        if ids is not None:
            ids = np.asarray(ids, dtype=np.int64)
            if not isinstance(self.index, faiss.IndexIDMap):
                self.index = faiss.IndexIDMap(self.index)
            self.index.add_with_ids(vectors, ids)
        else:
            self.index.add(vectors)
        
    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """ Search for the k nearest neighbors of the query vectors. """
        query_vectors = query_vectors.astype("float32")

        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # returns in order of similarity
        scores, indices = self.index.search(query_vectors, k)
        
        return scores, indices

    def save(self, path: str):
        """ Save the index to disk. """
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """ Load the index from disk. """
        self.index = faiss.read_index(path)
        