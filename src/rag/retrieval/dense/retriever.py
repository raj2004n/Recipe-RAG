from dataclasses import dataclass
from ..chunker import Chunk, ChunkSet
from .encoder import Encoder
from .indexer import Indexer


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
            self,
            chunk_sets: list[ChunkSet],
            encoder: Encoder,
            indexer: Indexer,
            top_k: int,
            score_threshold: float
        ):
        """ Initialise a Retriever object.

        Args:
            chunk_sets: ChunkSets this retriever can look up results against.
                        Sets that don't pass `accepts` are skipped — same
                        predicate must be used at build time so the indexer
                        only contains vectors for chunks the retriever has.
            encoder: Encoder used to encode the query.
            indexer: Indexer used to search for nearest neighbors.
            top_k: default number of results to return.
            score_threshold: drop results below this score.

        The encoder turns the query into a dense vector in the same space as the recipe vectors.
        The index is used to search for the nearest neighbors of the query vector (in the encoder space of the recipe vectors).
        The chunks maps the retrieved indices (top k) to the corresponding recipe chunks.
        """
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k
        self.score_threshold = score_threshold

        chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set.chunks]

        self.chunks_dict = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }


    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.top_k

        # encode query into dense vector
        query_vector = self.encoder.encode_query(query)

        # search recipe index for nearest neighbors of query vector
        scores, chunk_ids = self.indexer.search(query_vector, top_k)

        # build results
        results: list[RetrievalResult] = []

        for score, chunk_id in zip(scores[0], chunk_ids[0]):
            if chunk_id is None or chunk_id == -1:
                continue

            if self.score_threshold and score < self.score_threshold:
                continue

            # get chunk from dict
            chunk = self.chunks_dict.get(int(chunk_id))
            if chunk is None:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

            # break if enough results are found
            if len(results) >= top_k:
                break
        return results
