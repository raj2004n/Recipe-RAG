import faiss
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from .encoder import EncodedChunk

@dataclass
class IndexedChunkSet:
    name: str
    index: faiss.Index
    id_mapping: dict[int, int]

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

    def index_embeddings(self, encoded_chunks_sets: list[EncodedChunk]) -> list[IndexedChunkSet]:
        """ Assign ids to the embeddings. """
        # filter sets
        selected = [s for s in encoded_chunks_sets if self.accepts(s)]

        indexed_chunk_sets: list[IndexedChunkSet] = []
        for encoded_chunks_set in selected:
            vectors = np.asarray(
                [chunk.vector for chunk in encoded_chunks_set.encoded_chunks],
                dtype="float32"
            )

            faiss_ids = np.arange(len(encoded_chunks_set.encoded_chunks), dtype=np.int64)

            id_mapping: dict[int, int] = {
                int(faiss_id): chunk.chunk_id
                for faiss_id, chunk in zip(faiss_ids, encoded_chunks_set.encoded_chunks)
            }

            index = self._create_index()
            self._add(index, vectors, faiss_ids)

            indexed = IndexedChunkSet(
                name=encoded_chunks_set.name,
                index=index,
                id_mapping=id_mapping,
            )
            indexed_chunk_sets.append(indexed)

        # remember the indexed sets so search() can query across all of them
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
                "Indexer has no indexed sets — call index_embeddings() first."
            )

        query_vectors = query_vectors.astype("float32")
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)

        n_queries = query_vectors.shape[0]

        # collect (score, chunk_id) from every indexed set, then keep top k per query
        per_query_results: list[list[tuple[float, int]]] = [[] for _ in range(n_queries)]

        for indexed in self.indexed_sets:
            scores, indices = indexed.index.search(query_vectors, k)

            for q in range(n_queries):
                for score, faiss_idx in zip(scores[q], indices[q]):
                    if faiss_idx == -1:
                        continue
                    chunk_id = indexed.id_mapping.get(int(faiss_idx))
                    if chunk_id is None:
                        continue
                    per_query_results[q].append((float(score), chunk_id))

        # sort by score desc, take top-k, use -1 as the placeholder for missing slots
        out_scores = np.full((n_queries, k), -np.inf, dtype="float32")
        out_chunk_ids = np.full((n_queries, k), -1, dtype=np.int64)

        for q in range(n_queries):
            ranked = sorted(per_query_results[q], key=lambda x: x[0], reverse=True)[:k]
            for i, (score, chunk_id) in enumerate(ranked):
                out_scores[q, i] = score
                out_chunk_ids[q, i] = chunk_id

        return out_scores, out_chunk_ids

    # TODO: legacy single-set path — the build pipeline now writes per-set
    # indexes via save_indexed_chunk_sets. These stay for backward compat.
    def save(self, path: str):
        """ Save the index to disk. """
        if not self.indexed_sets:
            raise RuntimeError("No indexed sets to save.")
        # If there's only one set, persist in the legacy single-file format.
        if len(self.indexed_sets) == 1:
            indexed = self.indexed_sets[0]
            faiss.write_index(indexed.index, path)
            np.save(f"{path}.mapping.npy", indexed.id_mapping, allow_pickle=True)
            return

        raise RuntimeError(
            "save() only supports a single indexed set; multi-set builds use "
            "save_indexed_chunk_sets in core.snapshot instead."
        )

    @classmethod
    def load(cls, path: str, index_type: str, dimension: int) -> "Indexer":
        """ Load a single-set index from disk (legacy format). """
        indexer = cls(index_type, dimension)
        index = faiss.read_index(path)
        id_mapping = np.load(
            f"{path}.mapping.npy",
            allow_pickle=True,
        ).item()

        # wrap in a single IndexedChunkSet named after the file.
        name = Path(path).stem
        indexer.indexed_sets = [
            IndexedChunkSet(name=name, index=index, id_mapping=id_mapping)
        ]
        return indexer