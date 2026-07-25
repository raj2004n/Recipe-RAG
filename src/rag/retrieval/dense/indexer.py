import faiss
import numpy as np
from .encoder import EncodedChunkSet
from dataclasses import dataclass

@dataclass
class IndexedChunkSet:
    name: str
    index: faiss.Index
    id_mapping: dict[int, int]

class Indexer:
    def __init__(self, index_type: str, dimension: int):
        self.index_type = index_type
        self.dimension = dimension
        self.id_mapping: dict[int,str] = {}
    
    def _create_index(self) -> faiss.Index:
        """ Create a new index of the specified type. """
        if self.index_type == "FlatIP":
            return faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        
        raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _add(self, index: faiss.Index, vectors: np.ndarray, ids: np.ndarray):
        """ Add vectors to the index. """
        # numpy defualt is float64, but faiss requires float32
        vectors = vectors.astype("float32")
        ids = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(vectors, ids)

    def index_embeddings(self, encoded_chunks_sets: list[EncodedChunkSet]) -> list[IndexedChunkSet]:
        """ Assign ids to the embeddings. """
        indexed_chunk_sets = []
        for encoded_chunks_set in encoded_chunks_sets:
            vectors = np.asarray(
                [chunk.vector for chunk in encoded_chunks_set.encoded_chunks],
                dtype="float32"
            )

            # FAISS needs int ids
            faiss_ids = np.arange(len(encoded_chunks_set.encoded_chunks), dtype=np.int64)

            # keep mapping back to original chunk ids
            for faiss_id, chunk in zip(faiss_ids, encoded_chunks_set.encoded_chunks):
                self.id_mapping[int(faiss_id)] = chunk.chunk_id

            index = self._create_index()
            self._add(index, vectors, faiss_ids)

            indexed_chunk_sets.append(
                IndexedChunkSet(
                    name=encoded_chunks_set.name,
                    index=index,
                    id_mapping=self.id_mapping
                )
            )
        return indexed_chunk_sets

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

    # TODO old, replace with functions in snapshot
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