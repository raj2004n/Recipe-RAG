import faiss
import numpy as np
from dataclasses import dataclass
from .encoder import EncodedChunkSet


@dataclass
class IndexedChunkSet:
    name: str
    index: faiss.Index

    
class Indexer:
    def __init__(
        self,
        index_type: str,
        dimension: int
    ):
        self.index_type = index_type
        self.dimension = dimension
        self.indexed_sets: list[IndexedChunkSet] = []

    def _create_index(self) -> faiss.Index:
        """ Create a new index of the specified type. """
        if self.index_type == "FlatIP":
            return faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))

        raise ValueError(f"Unsupported index type: {self.index_type}")

    def _add(self, index: faiss.Index, vectors: np.ndarray, ids: np.ndarray):
        """ Add vectors to the index. """
        # numpy default is float64, but faiss requires float32
        vectors = vectors.astype("float32")
        ids = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(vectors, ids)


    def set_indexed_sets(self, indexed_sets: list[IndexedChunkSet]):
        self.indexed_sets = indexed_sets


    def index_embeddings(self, encoded_chunk_sets: list[EncodedChunkSet]) -> list[IndexedChunkSet]:
        """ Assign ids to the embeddings. """
        indexed_chunk_sets: list[IndexedChunkSet] = []

        for encoded_chunk_set in encoded_chunk_sets:
            chunks = encoded_chunk_set.encoded_chunks

            vectors = np.asarray(
                [chunk.vector for chunk in chunks],
                dtype="float32",
            )

            # use chunk IDs as FAISS IDs, since chunk ids are globally unique
            chunk_ids = np.asarray(
                [chunk.chunk_id for chunk in chunks],
                dtype=np.int64,
            )

            index = self._create_index()
            self._add(index, vectors, chunk_ids)

            indexed_chunk_sets.append(
                IndexedChunkSet(
                    name=encoded_chunk_set.name,
                    index=index,
                )
            )

        self.indexed_sets = indexed_chunk_sets

        return indexed_chunk_sets

    
    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """ Search across all indexed sets and merge top-k results.

        Returns:
            scores: np.ndarray of shape (n_queries, k) with similarity scores.
            chunk_ids: np.ndarray of shape (n_queries, k) with original chunk ids
                       (-1 where no result was returned).
        """
        if not self.indexed_sets:
            raise RuntimeError(
                "Indexer has no indexed sets — "
                "call index_embeddings() first."
            )

        query_vectors = np.asarray(
            query_vectors,
            dtype="float32",
        )

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        n_queries = query_vectors.shape[0]

        # collect (score, chunk_id) from every indexed set, then keep top k per query
        per_query_results: list[list[tuple[float, int]]] = [
            [] for _ in range(n_queries)
        ]

        for indexed in self.indexed_sets:
            scores, chunk_ids = indexed.index.search(
                query_vectors,
                k,
            )

            for q in range(n_queries):
                for score, chunk_id in zip(
                    scores[q],
                    chunk_ids[q],
                ):
                    if chunk_id == -1:
                        continue

                    per_query_results[q].append(
                        (float(score), int(chunk_id))
                    )

        # sort by score desc, take top-k, use -1 as the placeholder for missing slots
        out_scores = np.full(
            (n_queries, k),
            -np.inf,
            dtype="float32",
        )

        out_chunk_ids = np.full(
            (n_queries, k),
            -1,
            dtype=np.int64,
        )

        for q in range(n_queries):
            ranked = sorted(
                per_query_results[q],
                key=lambda x: x[0],
                reverse=True,
            )[:k]

            for i, (score, chunk_id) in enumerate(ranked):
                out_scores[q, i] = score
                out_chunk_ids[q, i] = chunk_id

        return out_scores, out_chunk_ids