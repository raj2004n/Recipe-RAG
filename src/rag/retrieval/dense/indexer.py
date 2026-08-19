import faiss
import numpy as np
from dataclasses import dataclass, field
from .encoder import EncodedChunkSet
from ..chunker import Chunk


# a FAISS id (chunk_id) maps back to its Chunk, so callers can resolve a
# hit to the underlying object without the indexer leaking ids
ChunkById = dict[int, Chunk]


@dataclass
class IndexedChunkSet:
    name: str
    index: faiss.Index
    chunks: ChunkById = field(default_factory=dict)
    # chunks is populated by index_embeddings (from EncodedChunk.chunk) or by
    # set_indexed_sets (from a snapshot's chunks.json) 
    # required for search() to resolve hits back to Chunks

    
class Indexer:
    def __init__(
        self,
        index_type: str,
        dimension: int,
        score_threshold: float | None = None,
    ):
        self.index_type = index_type
        self.dimension = dimension
        # Applied in every search() call. None disables the cutoff.
        self.score_threshold = score_threshold
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


    def set_indexed_sets(
        self,
        indexed_sets: list[IndexedChunkSet],
        chunks_by_set_name: dict[str, ChunkById] | None = None,
    ):
        """ Populate indexed_sets from pre-built indices (e.g. loaded from a snapshot).

        Args:
            indexed_sets: pre-built indices, one per chunk set.
            chunks_by_set_name: mapping of chunk-set name -> chunk_id -> Chunk.
                Used to attach the right chunks to each IndexedChunkSet so
                search() can resolve hits. Required for search() to work;
                omitted only when the caller doesn't intend to search.
        """
        if chunks_by_set_name is not None:
            for indexed in indexed_sets:
                indexed.chunks = chunks_by_set_name.get(indexed.name, {})

        self.indexed_sets = indexed_sets


    def search_one(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> list[tuple[float, Chunk | None]]:
        """Convenience wrapper around search() for the single-query case.

        Returns the top-k hits as (score, Chunk) tuples for one query, so
        callers don't have to unpack search()'s batched list-of-lists
        shape themselves. Equivalent to search(query_vector, k)[0] but
        raises IndexError if the underlying call somehow returns an empty
        list (i.e. no queries were processed), which would mask bugs.
        """
        results = self.search(query_vector, k)
        if not results:
            raise RuntimeError(
                "Indexer.search returned no results for a single query."
            )
        return results[0]


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

            # attach the source chunks so search() can resolve FAISS ids
            # back to a Chunk without anyone else holding a parallel dict.
            chunks_by_id: ChunkById = {
                enc.chunk_id: Chunk(
                    chunk_id=enc.chunk_id,
                    recipe_id=enc.recipe_id,
                    chunk_type=enc.chunk_type,
                    text=enc.text,
                )
                for enc in chunks
            }

            indexed_chunk_sets.append(
                IndexedChunkSet(
                    name=encoded_chunk_set.name,
                    index=index,
                    chunks=chunks_by_id,
                )
            )

        self.indexed_sets = indexed_chunk_sets

        return indexed_chunk_sets

    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> list[list[tuple[float, Chunk | None]]]:
        """ Search across all indexed sets and merge top-k results.

        Args:
            query_vectors: (n_queries, dim) or (dim,) dense query vectors.
            k: maximum number of hits to return per query.

        The indexer's score_threshold (set at construction) is applied after
        merging. Results are sorted descending, so a miss means the rest are
        also missed.

        Returns:
            A list of length n_queries; each inner list is the top-k hits for
            that query as (score, Chunk) tuples. Chunk is None when a hit's
            FAISS id doesn't resolve to a known chunk.
        """
        if not self.indexed_sets:
            raise RuntimeError(
                "Indexer has no indexed sets - "
                "call index_embeddings() first."
            )

        query_vectors = np.asarray(
            query_vectors,
            dtype="float32",
        )

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        n_queries = query_vectors.shape[0]

        # collect (score, chunk) from every indexed set, then keep top k per query
        per_query_results: list[list[tuple[float, Chunk | None]]] = [
            [] for _ in range(n_queries)
        ]

        for indexed in self.indexed_sets:
            scores, faiss_ids = indexed.index.search(
                query_vectors,
                k,
            )

            for q in range(n_queries):
                for score, faiss_id in zip(
                    scores[q],
                    faiss_ids[q],
                ):
                    if faiss_id == -1:
                        continue

                    chunk = indexed.chunks.get(int(faiss_id))
                    per_query_results[q].append(
                        (float(score), chunk)
                    )

        ranked_results: list[list[tuple[float, Chunk | None]]] = []
        for q in range(n_queries):
            ranked = sorted(
                per_query_results[q],
                key=lambda x: x[0],
                reverse=True,
            )[:k]

            if self.score_threshold is not None:
                # results are sorted descending, so truncate at the first miss
                truncated = []
                for score, chunk in ranked:
                    if score < self.score_threshold:
                        break
                    truncated.append((score, chunk))
                ranked = truncated

            ranked_results.append(ranked)

        return ranked_results